"""Source-owned, default-off sparse-attention observation hooks.

Importing this file imports no native runtime.  The caller supplies the already
owned StorageProvider and its single transfer/evidence ledgers.  All native
calls retain their original arguments, result and exception.  This is an
observation protocol, never a kernel selector or numerical acceptance rule.
"""
from dataclasses import dataclass, field
from functools import wraps, _lru_cache_wrapper
import hashlib
import inspect
import os
from pathlib import Path
import struct
import stat
import sys
import threading
from types import CodeType, FunctionType, MethodType, BuiltinFunctionType, ModuleType

from . import prefill_attention_plan as spec
from .prefill_attention_validation import (BACKEND_SHA, WRAPPER_SHA, frame_identity,
                                          semantic_contract)
from .prefill_diagnostic_plan import native_request_identity, remaining
from .prefill_storage import native_bytes, stored_row, add_failure_note

SOURCE_HASHES = {
    'vllm.v1.attention.backends.flashinfer': BACKEND_SHA,
    'vllm.model_executor.models.gemma4': '16ac0a67dcf5dd695a59edb177f20e1e69d9c3ec45c5883744470c7c06c516a3',
    'flashinfer.prefill': WRAPPER_SHA['fa2'],
    'flashinfer.decode': WRAPPER_SHA['xqa'],
    'flashinfer.xqa': '6a3f7330980b976201cff0fcdf51e2958e454c4c5951ad2b30c1f2aa13722d33',
    'flashinfer.utils': '6cb8ebcc25eb65521808bf40a8aaf0c5a868827283955867ae382ad7ec773815',
    'vllm.utils.flashinfer': '7144bbb7a905ad3d40994a2d3e69d69144e4e71020e7c92f487dea65b390790d',
    'flashinfer.api_logging': '0302b4ff890c9d0de9c91f340794d8805b68b467690fe3fbec3dfb29f05720ad',
    'flashinfer.trace.template': '8188ab3ff4be944bc15817f10da342ea9e0c7e25a4a29f8868140e5e685c6825',
}


def tensor_identity(tensor, *, object_identity=True):
    storage = tensor.untyped_storage()
    fields = (storage.data_ptr(), storage.nbytes(), tensor.storage_offset(),
              tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype), str(tensor.device))
    return (id(tensor), *fields) if object_identity else fields


def _closure(function):
    return dict(zip(function.__code__.co_freevars,
                    (cell.cell_contents for cell in (function.__closure__ or ()))))


class SourceGuard:
    """Bind loaded module/file/code and the reviewed zero-dump wrapper closure.

    A matching adjacent file or spoofed __wrapped__/__module__ is insufficient:
    each executing Python code object must occur in compilation of those exact
    admitted bytes. Compilation executes no imports or native code.
    """
    def __init__(self, installed_sources):
        new_sources = {'vllm.utils.flashinfer', 'flashinfer.api_logging',
                       'flashinfer.trace.template', 'flashinfer.utils'}
        if (set(installed_sources) != new_sources
                or any(installed_sources.get(name) != SOURCE_HASHES[name] for name in new_sources)):
            raise RuntimeError('Complete installed attention source closure not admitted')
        self.expected = dict(SOURCE_HASHES)
        self.modules, self.files, self.codes = {}, {}, {}
        self.bindings = []
        self.method_owners = []
        self.aliases, self.owned_hooks, self.enums = [], [], []
        self.runtime_objects = []
        # Only inspect modules the backend has already imported. Never import
        # or resolve a different native path to make capture admission succeed.
        for name in self.expected:
            self.module(name)

    def module(self, name):
        module = sys.modules.get(name)
        if module is None or getattr(module, '__name__', None) != name:
            raise RuntimeError('Actual attention module is not loaded: ' + name)
        path = Path(getattr(module, '__file__', '')).absolute()
        if (path.is_symlink() or any(p.is_symlink() for p in path.parents)
                or path.suffix != '.py'
                or getattr(getattr(module, '__spec__', None), 'origin', None) != str(path)):
            raise RuntimeError('Actual source module origin must be regular Python source')
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 1 << 20:
                raise RuntimeError('Bounded regular attention source required')
            chunks, left = [], before.st_size + 1
            while left:
                chunk = os.read(fd, min(left, 65536))
                if not chunk:
                    break
                chunks.append(chunk); left -= len(chunk)
            raw = b''.join(chunks)
            after = os.fstat(fd)
            identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
            if len(raw) != before.st_size or identity(before) != identity(after):
                raise RuntimeError('Attention source changed during bounded read')
        finally:
            os.close(fd)
        if hashlib.sha256(raw).hexdigest() != self.expected[name]:
            raise RuntimeError('Actual attention source bytes differ: ' + name)
        if name in self.modules and (self.modules[name] is not module or self.files[name] != path):
            raise RuntimeError('Attention source module/origin replaced: ' + name)
        self.modules[name], self.files[name] = module, path
        if name not in self.codes:
            def descend(code):
                yield code
                for value in code.co_consts:
                    if isinstance(value, CodeType):
                        yield from descend(value)
            self.codes[name] = tuple(descend(compile(raw, str(path), 'exec',
                                                    dont_inherit=True, optimize=sys.flags.optimize)))
        return module

    def function(self, function, name):
        module = self.module(name)
        if (type(function) is not FunctionType or function.__globals__ is not vars(module)
                or Path(function.__code__.co_filename).absolute() != self.files[name]
                or function.__code__ not in self.codes[name]):
            raise RuntimeError('Attention callable is not its admitted executing source: ' + name)
        return function

    def callable(self, function, module, qualname):
        chain = []
        current = function
        for _ in range(4):
            if type(current) is not FunctionType or current in chain:
                raise RuntimeError('Unsupported attention callable/decorator chain')
            chain.append(current)
            wrapped = getattr(current, '__wrapped__', None)
            if wrapped is None:
                self.function(current, module)
                if current.__qualname__ != qualname:
                    raise RuntimeError('Attention callable source owner changed')
                break
            self.function(current, 'flashinfer.api_logging')
            if current.__code__.co_name != '_auto_dump_wrapper':
                raise RuntimeError('Only the disabled trace-forwarding decorator is admitted')
            closure = _closure(current)
            if closure.get('_inner') is not wrapped:
                raise RuntimeError('Decorator __wrapped__ differs from actual callee')
            helper = closure.get('_is_trace_dump_enabled')
            self.function(helper, 'flashinfer.trace.template')
            if helper.__qualname__ != '_is_trace_dump_enabled':
                raise RuntimeError('Unknown trace-enable closure')
            current = wrapped
        else:
            raise RuntimeError('Too many attention wrapper layers')
        logging = self.module('flashinfer.api_logging')
        if (logging._API_LOG_LEVEL != 0 or os.environ.get('FLASHINFER_TRACE_DUMP', '0') not in ('0', '')
                or os.environ.get('FLASHINFER_LOGLEVEL', '0') != '0'):
            raise RuntimeError('Attention logging/trace dumping is outside the evidence budget')
        snapshot = tuple(chain)
        if not any(entry[0] is function for entry in self.bindings):
            self.bindings.append((function, module, qualname, snapshot))
        return current

    def method(self, owner, name, module, cls, *, watch=True):
        loaded = self.module(module)
        function = getattr(owner, name)
        if (type(owner) is not getattr(loaded, cls, None) or name in vars(owner)
                or type(function) is not MethodType or function.__self__ is not owner
                or function.__func__ is not getattr(type(owner), name)):
            raise RuntimeError('Actual attention method/class owner differs')
        self.callable(function.__func__, module, cls + '.' + name)
        if watch and not any(entry[0] is owner and entry[1] == name for entry in self.method_owners):
            self.method_owners.append((owner, name, function.__func__))
        return function

    def alias(self, consumer, attribute, module, name=None):
        """Bind source-defined helper, cached helper or exact reviewed enum."""
        imported = getattr(consumer, attribute)
        source = self.module(module)
        exported = getattr(source, name or attribute, None)
        if imported is not exported:
            raise RuntimeError('Attention helper alias differs from source export: ' + attribute)
        if isinstance(imported, type):
            from enum import EnumMeta
            if not isinstance(imported, EnumMeta) or imported.__module__ != module:
                raise RuntimeError('Unsupported attention semantic enum owner')
            expected = {'MaskMode': {'NON_CAUSAL':0,'CAUSAL':1,'CUSTOM':2,'MULTIITEMSCORING':3},
                        'TensorLayout': {'NHD':0,'HND':1},
                        'PosEncodingMode': {'NONE':0,'ROPE_LLAMA':1,'ALIBI':2}}[attribute]
            if {k:v.value for k,v in imported.__members__.items()} != expected:
                raise RuntimeError('Attention semantic enum values changed')
            if not any(enum is imported for enum,_ in self.enums):
                self.enums.append((imported,expected))
        elif type(imported) is _lru_cache_wrapper:
            self.callable(imported.__wrapped__, module, name or attribute)
            if imported.__wrapped__.__qualname__ != (name or attribute):
                raise RuntimeError('Unknown cached attention helper source')
        elif getattr(imported, '__wrapped__', None) is not None:
            self.callable(imported, module, name or attribute)
        else:
            self.callable(imported, module, name or attribute)
            if imported.__qualname__ != (name or attribute):
                raise RuntimeError('Attention helper source owner changed')
        if not any(owner is consumer and attr == attribute for owner,attr,_ in self.aliases):
            self.aliases.append((consumer, attribute, imported))
        return imported

    def native_leaf(self, callback):
        """Reject a Python replacement for the already loaded opaque kernel.

        This identifies the extension origin, not numerical rounding or build
        equivalence. Unknown extension interfaces remain fail closed.
        """
        if type(callback) is BuiltinFunctionType:
            module = sys.modules.get(callback.__module__)
            if not isinstance(module, ModuleType) or getattr(module, callback.__name__, None) is not callback:
                raise RuntimeError('Native attention export has no actual loaded module owner')
        else:
            cls = type(callback)
            module = sys.modules.get(cls.__module__)
            if (cls.__module__ != 'tvm_ffi.core' or cls.__name__ != 'Function'
                    or module is None or getattr(module, 'Function', None) is not cls
                    or isinstance(cls.__call__, FunctionType)):
                raise RuntimeError('Foreign Python or unknown native attention producer')
        path = Path(getattr(module, '__file__', '')).absolute()
        if (path.is_symlink() or any(p.is_symlink() for p in path.parents)
                or '.so' not in path.suffixes
                or getattr(getattr(module, '__spec__', None), 'origin', None) != str(path)):
            raise RuntimeError('Native attention extension origin unsupported')
        st = path.stat()
        if not stat.S_ISREG(st.st_mode):
            raise RuntimeError('Native attention extension is not regular')
        return {'module':module.__name__, 'origin':str(path), 'bytes':st.st_size,
                'device':st.st_dev, 'inode':st.st_ino, 'mtime_ns':st.st_mtime_ns,
                'callable_type':type(callback).__module__+'.'+type(callback).__name__}

    def require_current(self):
        for owner,name,value in self.runtime_objects:
            if getattr(owner,name,None) is not value:
                raise RuntimeError('Attention runtime module object replaced: ' + name)
        for enum,expected in self.enums:
            if {k:v.value for k,v in enum.__members__.items()} != expected:
                raise RuntimeError('Attention semantic enum values changed')
        for consumer, attribute, original in self.aliases:
            actual = getattr(consumer, attribute, None)
            if actual is not original:
                owned = [h for h in self.owned_hooks if h.active and h.owner is consumer
                         and h.name == attribute and h.original is original]
                if len(owned) != 1:
                    raise RuntimeError('Attention source helper replaced: ' + attribute)
                owned[0].require_current()
        for owner, name, original in self.method_owners:
            current = getattr(owner, name, None)
            if (name in vars(owner) or type(current) is not MethodType
                    or current.__self__ is not owner or current.__func__ is not original
                    or getattr(type(owner), name, None) is not original):
                raise RuntimeError('Attention source-owned method replaced: ' + name)
        for name in tuple(self.modules):
            self.module(name)
        for function, module, qualname, chain in tuple(self.bindings):
            self.callable(function, module, qualname)
            current = function
            actual = []
            while current is not None:
                actual.append(current)
                current = getattr(current, '__wrapped__', None)
                if len(actual) > 4:
                    raise RuntimeError('Attention decorator chain changed')
            if tuple(actual) != chain:
                raise RuntimeError('Attention decorator function replacement')


class ObservedCallHook:
    """One owned attribute; unchanged forwarding, including BaseException.

    before/after/finally observers never replace the primary native exception.
    Unlike an unscoped class patch, a bound hook belongs to one selected object.
    """
    def __init__(self, owner, name, before, after, fail, *, final=None, signature=None):
        self.owner, self.name, self.original = owner, name, getattr(owner, name)
        self.bound = type(self.original) is MethodType
        self.had_attribute = name in vars(owner)
        if self.bound and self.had_attribute:
            raise RuntimeError('Preexisting attention instance override')
        self.class_callback = getattr(type(owner), name, None) if self.bound else None
        self.signature = signature or inspect.signature(self.original)
        self.active = True

        @wraps(self.original)
        def invoke(*args, **kwargs):
            ticket, primary = None, None
            try:
                self.require_current()
                bound = self.signature.bind(*args, **kwargs)
                bound.apply_defaults()
                ticket = before(bound.arguments)
                result = self.original(*args, **kwargs)
                after(ticket, result)
                return result
            except BaseException as error:
                primary = error
                try:
                    fail(error)
                except BaseException as secondary:
                    add_failure_note(error, 'Secondary attention failure handler: ' + repr(secondary))
                raise
            finally:
                if final is not None:
                    try:
                        final(ticket, primary)
                    except BaseException as error:
                        if primary is not None:
                            add_failure_note(primary, 'Secondary attention finalization: ' + repr(error))
                        else:
                            try:
                                fail(error)
                            except BaseException as secondary:
                                add_failure_note(error, 'Secondary attention failure handler: ' + repr(secondary))
                            raise

        if self.bound:
            @wraps(self.original)
            def wrapper(actual, *args, **kwargs):
                if actual is not owner:
                    raise RuntimeError('Attention selected method receiver changed')
                return invoke(*args, **kwargs)
            self.wrapper = wrapper
            setattr(owner, name, MethodType(wrapper, owner))
        else:
            self.wrapper = invoke
            setattr(owner, name, invoke)

    def require_current(self):
        actual = getattr(self.owner, self.name, None)
        okay = (type(actual) is MethodType and actual.__self__ is self.owner
                and actual.__func__ is self.wrapper
                and getattr(type(self.owner), self.name) is self.class_callback) if self.bound else actual is self.wrapper
        if not self.active or not okay:
            raise RuntimeError('Attention hook owner/function replaced: ' + self.name)

    def restore(self):
        if not self.active:
            return
        self.require_current()
        if self.bound:
            delattr(self.owner, self.name)
        else:
            setattr(self.owner, self.name, self.original)
        self.active = False


@dataclass
class AttentionTicket:
    layer: int
    frame: object
    start: int
    end: int
    arguments: dict
    query_identity: tuple
    output_identity: tuple
    cache_identity: tuple
    thread: int
    entrypoint: str
    wrapper: object = None
    hook: object = None
    entered: bool = False
    complete: bool = False
    binding: dict = field(default_factory=dict)
    tensors: list = field(default_factory=list)
    record: dict = field(default_factory=dict)
    resolver_hook: object = None
    kernel_seen: bool = False
    native_callback: object = None
    policy: object = None
    metadata_reads: list = field(default_factory=list)


class AttentionHooks:
    """Attach to an admitted StorageProvider for the exact twelve operations.

    API: processed_row; require_current; require_frame_complete;
    require_complete; restore; records. Evidence must provide head_row and
    read_selection. The caller precharges attention_metadata = 65,536 bytes.
    """
    def __init__(self, provider, *, installed_sources):
        self.provider, self.torch = provider, provider.torch
        self.guard = SourceGuard(installed_sources)
        self.hooks, self.records, self.seen, self.writer_rows = [], [], set(), {0: {}, 5: {}}
        self.active_ticket, self.failed, self.restored = None, False, False
        self.drain_failed = False
        self.metadata_bytes = 0
        try:
            self._bind_helpers()
            for layer in spec.LAYERS:
                _, parent, _, impl = provider.layers[layer]
                self.guard.method(parent, 'forward', 'vllm.model_executor.models.gemma4', 'Gemma4Attention')
                for name in ('forward', 'maybe_quant_query', 'get_xqa_bmm1_scale'):
                    self.guard.method(impl, name, 'vllm.v1.attention.backends.flashinfer', 'FlashInferImpl',
                                      watch=name != 'forward')
                self.hooks.append(ObservedCallHook(impl, 'forward',
                    lambda args, layer=layer: self._begin(layer, args), self._end, self._fail,
                    final=self._final))
        except BaseException as primary:
            self.restore(primary)
            raise

    def _bind_helpers(self):
        for name in ('flashinfer.prefill','flashinfer.decode','flashinfer.utils','flashinfer.xqa',
                     'vllm.v1.attention.backends.flashinfer'):
            module=self.guard.module(name)
            if module.torch is not self.torch:
                raise RuntimeError('Attention source uses a different actual Torch runtime')
            self.guard.runtime_objects.append((module,'torch',self.torch))
        utils = self.guard.module('flashinfer.utils')
        for name in ('_expand_4d','_expand_5d','get_compute_capability','get_device_properties',
                     'get_alibi_slopes','PosEncodingMode','MaskMode','TensorLayout',
                     '_unpack_paged_kv_cache','_check_cached_qkv_data_type',
                     'device_support_pdl','is_float8','_get_cache_alibi_slopes_buf',
                     'check_shape_dtype_device','check_trtllm_gen_sm107_only_feature',
                     '_check_pos_encoding_mode','get_device_sm_count'):
            self.guard.alias(utils,name,'flashinfer.utils')
        for module in ('flashinfer.prefill','flashinfer.decode'):
            owner = self.guard.module(module)
            for name in ('_unpack_paged_kv_cache','_check_cached_qkv_data_type',
                         'device_support_pdl','is_float8','_get_cache_alibi_slopes_buf',
                         'check_shape_dtype_device','MaskMode','TensorLayout'):
                self.guard.alias(owner,name,'flashinfer.utils')
            if module == 'flashinfer.prefill':
                self.guard.alias(owner,'check_trtllm_gen_sm107_only_feature','flashinfer.utils')
                self.guard.alias(owner,'_split_scale_param','flashinfer.prefill')
            else:
                self.guard.alias(owner,'_check_pos_encoding_mode','flashinfer.utils')
                self.guard.alias(owner,'get_device_sm_count','flashinfer.utils')
                self.guard.alias(owner,'xqa','flashinfer.xqa')
        xqa = self.guard.module('flashinfer.xqa')
        for name in ('get_compute_capability','get_device_sm_count','device_support_pdl'):
            self.guard.alias(xqa,name,'flashinfer.utils')
        for name in ('get_xqa_module','_get_xqa_module_cached'):
            self.guard.alias(xqa,name,'flashinfer.xqa')

    def processed_row(self, layer, position, slot, key, value):
        if layer not in spec.LAYERS:
            return
        if (self.failed or self.restored or type(position) is not int or not 0 <= position <= 2048
                or position in self.writer_rows[layer]):
            raise RuntimeError('Duplicate, out-of-range or poisoned attention writer contribution')
        selected = spec.select_writer_row(layer, position, key, value)
        self.writer_rows[layer][position] = (slot, *(hashlib.sha256(data).digest() for _, _, data in selected))

    def _fail(self, primary):
        self.failed = True
        self.provider.failed = True
        # StorageProvider owns failure receipt, ledger abort and original hooks.
        self.provider.poison_storage(primary)

    def _tensor(self, value, shape=None):
        if (not isinstance(value, self.torch.Tensor) or value.dtype != self.torch.bfloat16
                or value.device.type != 'cuda' or (shape is not None and tuple(value.shape) != shape)
                or any(s < 0 or (s == 0 and n != 1) for n,s in zip(value.shape,value.stride()))
                or value.stride()[-1] != 1):
            raise RuntimeError('Selected attention BF16 CUDA tensor geometry differs')
        return tensor_identity(value)

    def _same_region(self, actual, expected):
        def region(tensor):
            identity = tensor_identity(tensor, object_identity=False)
            # A size-one dimension has only index zero; its stride cannot
            # change any consumed address. The native source canonicalizes it.
            return (*identity[:4], tuple(s if n != 1 else 0 for n,s in zip(tensor.shape,tensor.stride())), *identity[5:])
        if region(actual) != region(expected):
            raise RuntimeError('Actual attention Q/out does not alias enclosing original region')

    def _begin(self, layer, arguments):
        p = self.provider
        if not p.started or p.completed:
            return None
        if self.failed or p.failed or self.restored or self.active_ticket is not None:
            raise RuntimeError('Poisoned or reentrant selected attention forward')
        self.require_current()
        frame = p.frame
        if (frame is None or frame.owner is not p or frame.sequence != p.ledger.frames
                or p.ledger.pending is None):
            raise RuntimeError('Missing current enclosing attention frame')
        start, end, _ = p.ledger.pending
        if (start, end) not in spec.FRAMES:
            return None
        if (start, end, layer) in self.seen or layer not in p.writer_seen:
            raise RuntimeError('Duplicate attention call or missing completed native writer')
        name, parent, attn, impl = p.layers[layer]
        meta = arguments['attn_metadata']
        d, heads = spec.LAYERS[layer]['dim'], spec.LAYERS[layer]['kv_heads']
        if (arguments['layer'] is not attn or arguments['kv_cache'] is not attn.kv_cache
                or meta is not frame.context.attn_metadata[name] or parent.attn is not attn or attn.impl is not impl
                or frame.context.no_compile_layers[name] is not attn
                or arguments['output_scale'] is not None or arguments['output_block_scale'] is not None
                or impl.dcp_world_size != 1 or impl.sinks is not None or impl.is_kvcache_nvfp4
                or impl.kv_sharing_target_layer_name is not None or attn.kv_sharing_target_layer_name is not None
                or parent.is_kv_shared_layer is not False or impl.scale != 1.0
                or impl.logits_soft_cap not in (None, 0.0) or impl.window_left != (1023 if layer == 0 else -1)
                or impl.num_heads != 16 or impl.num_kv_heads != heads or impl.head_size != d
                or impl.kv_cache_dtype not in ('auto', 'bfloat16')
                or any(type(getattr(attn, k)) not in (int, float) or getattr(attn, k) != 1.0
                       for k in ('_q_scale_float', '_k_scale_float', '_v_scale_float'))
                or meta.use_cascade or meta.causal is not True or meta.num_actual_tokens != end-start
                or meta.q_data_type_prefill != self.torch.bfloat16
                or meta.q_data_type_decode != self.torch.bfloat16):
            raise RuntimeError('Enclosing attention owner/semantic/dtype policy differs')
        query, output = arguments['query'], arguments['output']
        qi, oi = self._tensor(query, (end-start, 16, d)), self._tensor(output, (end-start, 16, d))
        ci = self._tensor(attn.kv_cache)
        owned = (id(attn.kv_cache), ci[1], ci[2], ci[3], ci[4], ci[5], ci[7])
        if owned != frame.identities[layer] or qi[1] == oi[1] or qi[1] == ci[1] or oi[1] == ci[1]:
            raise RuntimeError('Attention allocation/role alias changed')
        backend = self.guard.module('vllm.v1.attention.backends.flashinfer')
        if end-start == 256:
            if (meta.num_prefills != 1 or meta.num_prefill_tokens != 256 or meta.num_decodes != 0
                    or meta.num_decode_tokens != 0 or type(meta.prefill) is not backend.FIPrefill):
                raise RuntimeError('Unreviewed prefill dispatch')
            route, wrapper = 'paged_prefill', meta.prefill.wrapper
        elif type(meta.decode) is backend.FIDecode:
            route, wrapper = 'paged_decode', meta.decode.wrapper
        elif (type(meta.decode) is backend.FlashInferTrtllmAPIDecode
              and meta.decode.kernel is backend.FlashInferDecodeKernel.XQA and layer == 0):
            route, wrapper = 'xqa_decode', None
        else:
            raise RuntimeError('Unknown decode route or forbidden global D512 XQA')
        if end-start == 1 and (meta.num_decodes != 1 or meta.num_decode_tokens != 1
                              or meta.num_prefills != 0 or meta.num_prefill_tokens != 0):
            raise RuntimeError('Mixed or padded decode request')
        ticket = AttentionTicket(layer, frame, start, end, dict(arguments), qi, oi, ci,
                                 threading.get_ident(), route, wrapper)
        self.active_ticket = ticket
        try:
            if wrapper is not None:
                module, cls = ('flashinfer.prefill', 'BatchPrefillWithPagedKVCacheWrapper') if route == 'paged_prefill' else ('flashinfer.decode', 'BatchDecodeWithPagedKVCacheWrapper')
                self.guard.method(wrapper, 'run', module, cls, watch=False)
                ticket.hook = ObservedCallHook(wrapper, 'run', lambda args: self._operator_before(ticket, args),
                                              self._operator_after, self._fail)
            else:
                alias = self._lazy_alias()
                inner = self._cached_xqa(alias)
                ticket.binding['lazy_alias'] = id(alias)
                ticket.binding['lazy_inner'] = id(inner)
                xqa_module = self.guard.module('flashinfer.xqa')
                ticket.resolver_hook = ObservedCallHook(xqa_module, 'get_xqa_module',
                    lambda args: self._xqa_resolve_before(ticket,args),
                    self._xqa_resolve_after, self._fail)
                self.guard.owned_hooks.append(ticket.resolver_hook)
                ticket.hook = ObservedCallHook(backend, 'flashinfer_xqa_batch_decode_with_kv_cache',
                    lambda args: self._operator_before(ticket, args), self._operator_after, self._fail,
                    signature=inspect.signature(inner))
        except BaseException:
            # A resolver may already be installed. Keep this ticket reachable
            # for the primary-error-preserving centralized drain/restoration.
            raise
        return ticket

    def _lazy_alias(self):
        shim = self.guard.module('vllm.utils.flashinfer')
        backend = self.guard.module('vllm.v1.attention.backends.flashinfer')
        alias = backend.flashinfer_xqa_batch_decode_with_kv_cache
        if alias is not shim.flashinfer_xqa_batch_decode_with_kv_cache:
            raise RuntimeError('Backend XQA alias differs from actual lazy shim')
        self.guard.function(alias, 'vllm.utils.flashinfer')
        if alias.__qualname__ != '_lazy_import_wrapper.<locals>.wrapper':
            raise RuntimeError('Unknown XQA forwarding closure')
        return alias

    def _cached_xqa(self, alias):
        closure = _closure(alias)
        getter = closure.get('_get_impl')
        if type(getter) is not _lru_cache_wrapper or getter.cache_info().currsize != 1:
            raise RuntimeError('XQA lazy implementation not already naturally resolved')
        self.guard.function(getter.__wrapped__, 'vllm.utils.flashinfer')
        captured = _closure(getter.__wrapped__)
        if (captured != {'module_name': 'flashinfer.decode', 'attr_name': 'xqa_batch_decode_with_kv_cache'}
                or getter.__wrapped__.__qualname__ != '_lazy_import_wrapper.<locals>._get_impl'):
            raise RuntimeError('XQA cached module/attribute closure differs')
        before = getter.cache_info()
        inner = getter()  # cache hit only; never clear, cold-resolve, or substitute
        after = getter.cache_info()
        if after.misses != before.misses or after.currsize != 1 or after.hits != before.hits+1:
            raise RuntimeError('XQA lazy resolver was not an existing cache hit')
        decode = self.guard.module('flashinfer.decode')
        if inner is not decode.xqa_batch_decode_with_kv_cache:
            raise RuntimeError('Cached XQA callee differs from source-owned public export')
        self.guard.callable(inner, 'flashinfer.decode', 'xqa_batch_decode_with_kv_cache')
        return inner

    def _metadata(self, tensor, shape, expected, *, record=True):
        if (not isinstance(tensor, self.torch.Tensor) or tuple(tensor.shape) != shape
                or tensor.dtype != self.torch.int32 or tensor.device.type != 'cuda'):
            raise RuntimeError('Actual operator metadata extent/dtype/device differs')
        size = tensor.numel()*4
        if self.metadata_bytes+size > 65536:
            raise RuntimeError('Actual attention metadata exceeds precharged allowance')
        self.metadata_bytes += size
        values = tensor.detach().cpu().tolist()
        if values != expected:
            raise RuntimeError('Actual operator table/length differs from original manager inputs')
        if record:
            self.active_ticket.metadata_reads.append((tensor,shape,expected))
        return tensor_identity(tensor)

    def _policy(self,ticket):
        p=self.provider
        _,parent,attn,impl=p.layers[ticket.layer]
        meta=ticket.arguments['attn_metadata']
        def attrs(owner,names):
            result=[]
            for name in names:
                value=getattr(owner,name)
                if isinstance(value,self.torch.Tensor):
                    value=tensor_identity(value)
                elif value is not None and type(value) not in (str,int,float,bool):
                    value=(id(value),str(value) if value in (self.torch.bfloat16,) else type(value).__name__)
                result.append((name,value))
            return result
        result={'impl':attrs(impl,('num_heads','num_kv_heads','head_size','dcp_world_size',
            'sinks','is_kvcache_nvfp4','kv_sharing_target_layer_name','scale','logits_soft_cap',
            'window_left','kv_cache_dtype')),
            'attn':attrs(attn,('_q_scale_float','_k_scale_float','_v_scale_float',
                              'kv_sharing_target_layer_name')),
            'parent':attrs(parent,('is_kv_shared_layer',)),
            'metadata':attrs(meta,('q_data_type_prefill','q_data_type_decode','num_actual_tokens',
                'num_decodes','num_decode_tokens','num_prefills','num_prefill_tokens','causal','use_cascade')),
            'manager':(id(p.manager),p.manager.receipt()),
            'layout':tuple(impl.kv_cache_layout.layer_view_order)}
        if ticket.wrapper is not None:
            result['wrapper']=attrs(ticket.wrapper,('_backend','_jit_module','_use_cuda_graph',
                '_kv_layout','_pos_encoding_mode','_sm_scale','_logits_soft_cap','_window_left',
                '_cached_q_data_type','_cached_kv_data_type','_cached_o_data_type','_num_qo_heads',
                '_num_kv_heads','_paged_kv_indptr_buf','_paged_kv_indices_buf','_paged_kv_last_page_len_buf',
                '_qo_indptr_buf','_rope_scale','_rope_theta'))
            if ticket.entrypoint=='paged_prefill':
                result['prefill']=attrs(ticket.wrapper,('_causal','_use_fp16_qk_reduction',
                    '_custom_mask_buf','_mask_indptr_buf','_prefix_len_ptr','_token_pos_in_items_ptr',
                    '_max_item_len_ptr','_token_pos_in_items_len','_qo_indptr_last'))
            else:
                result['decode']=attrs(ticket.wrapper,('_q_len_per_req','_window_right','_use_tensor_cores'))
        else:
            result['xqa']=attrs(meta.decode,('block_tables','seq_lens','max_seq_len',
                                             'q_len_per_req','mask','q_cu_seq_lens','kernel'))
        return result

    def _check_ticket(self, ticket):
        p = self.provider
        if (self.failed or p.failed or self.active_ticket is not ticket or p.frame is not ticket.frame
                or threading.get_ident() != ticket.thread or p.ledger.pending is None
                or p.ledger.pending[:2] != (ticket.start, ticket.end)
                or ticket.frame.sequence != p.ledger.frames):
            raise RuntimeError('Stale, foreign-thread or poisoned attention ticket')
        for key, expected in (('query', ticket.query_identity), ('output', ticket.output_identity),
                              ('kv_cache', ticket.cache_identity)):
            if tensor_identity(ticket.arguments[key]) != expected:
                raise RuntimeError('Enclosing attention tensor identity drift')
        if ticket.hook is not None:
            ticket.hook.require_current()
        self.guard.require_current()

    def _xqa_resolve_before(self,ticket,args):
        self._check_ticket(ticket)
        if ticket.kernel_seen:
            raise RuntimeError('Repeated XQA module resolution inside one call')
        expected = {'input_dtype':self.torch.bfloat16,'kv_cache_dtype':self.torch.bfloat16,
            'page_size':16,'head_dim':256,'head_group_ratio':2,'use_sliding_window':True,
            'output_dtype':self.torch.bfloat16,'q_seq_len':1,'use_ragged_q':False}
        if args != expected:
            raise RuntimeError('Actual XQA resolved kernel geometry differs')
        return ticket

    def _xqa_resolve_after(self,ticket,result):
        self._check_ticket(ticket)
        callback = getattr(result,'xqa',None)
        self.guard.function(callback,'flashinfer.xqa')
        closed = _closure(callback)
        if callback.__qualname__ != '_get_xqa_module_cached.<locals>.xqa' or set(closed) != {'module'}:
            raise RuntimeError('XQA selected native wrapper source/closure differs')
        native = getattr(closed['module'],'xqa_wrapper',None)
        ticket.native_callback = native
        ticket.binding['native_origin'] = self.guard.native_leaf(native)
        ticket.binding['native_callee'] = id(native)
        ticket.binding['xqa_kernel'] = id(callback)
        ticket.binding['xqa_native'] = id(native)
        ticket.kernel_seen = True

    def _wrapper_plan(self, ticket, args):
        wrapper, layer = ticket.wrapper, ticket.layer
        if (wrapper._backend != 'fa2' or wrapper._jit_module is not None or wrapper._use_cuda_graph
                or wrapper._kv_layout not in ('HND', 'NHD')
                or wrapper._pos_encoding_mode != 'NONE' or wrapper._sm_scale != 1.0
                or wrapper._logits_soft_cap not in (None, 0.0)
                or wrapper._window_left != (1023 if layer == 0 else -1)
                or wrapper._cached_q_data_type != self.torch.bfloat16
                or wrapper._cached_kv_data_type != self.torch.bfloat16
                or wrapper._cached_o_data_type != self.torch.bfloat16
                or wrapper._num_qo_heads != 16 or wrapper._num_kv_heads != spec.LAYERS[layer]['kv_heads']
                or args.get('args', ()) or any(args.get(k) is not None for k in
                   ('lse','sinks','kv_cache_sf','skip_softmax_threshold_scale_factor','use_fp16_softmax','uses_spcompress'))
                or args['return_lse'] is not False
                or args.get('window_left') not in (None, wrapper._window_left)
                or any(type(args[k]) not in (int,float) or args[k] != 1.0 for k in ('q_scale','k_scale','v_scale'))):
            raise RuntimeError('Actual wrapper plan/scales/mask/dtype/dispatch unsupported')
        if ticket.entrypoint == 'paged_prefill':
            if (wrapper._causal is not True or wrapper._use_fp16_qk_reduction
                    or any(getattr(wrapper, k) is not None for k in
                           ('_custom_mask_buf','_mask_indptr_buf','_prefix_len_ptr','_token_pos_in_items_ptr','_max_item_len_ptr'))
                    or wrapper._token_pos_in_items_len != 0 or wrapper._qo_indptr_last != ticket.end-ticket.start):
                raise RuntimeError('Prefill mask, positions or reduction differs')
        elif (wrapper._q_len_per_req != 1 or args.get('q_len_per_req') not in (None,1)
              or wrapper._window_right != -1):
            raise RuntimeError('Paged decode length/window differs')
        plan = wrapper._plan_info
        if type(plan) not in (list,tuple) or not 0 < len(plan) <= 128 or any(type(v) is not int for v in plan):
            raise RuntimeError('Missing bounded actual native split plan')
        kernel = wrapper._cached_module
        name = 'paged_run' if ticket.entrypoint == 'paged_prefill' or wrapper.use_tensor_cores else 'run'
        callback = getattr(kernel, name, None)
        if ticket.entrypoint == 'paged_decode' and wrapper.use_tensor_cores is not True:
            raise RuntimeError('Actual backend native decode tensor-core route changed')
        self.guard.function(callback, 'flashinfer.prefill')
        closed = _closure(callback)
        if (callback.__qualname__ != 'get_batch_prefill_module.<locals>.paged_run'
                or set(closed) != {'backend','paged_run_func'} or closed['backend'] != 'fa2'):
            raise RuntimeError('Resolved paged-run producer source/closure differs')
        native = closed['paged_run_func']
        ticket.native_callback = native
        ticket.binding['native_origin'] = self.guard.native_leaf(native)
        ticket.binding['native_callee'] = id(native)
        # Kernel/binary numerical identity remains unqualified, but preserve
        # its actual owner/callable and actual split-plan values across the call.
        ticket.binding.update(wrapper=id(wrapper), kernel=id(kernel), kernel_callback=id(callback),
                              kernel_name=name, split_plan=list(plan), split_plan_owner=id(plan),
                              layout=wrapper._kv_layout,
                              use_tensor_cores=getattr(wrapper, 'use_tensor_cores', None))
        return wrapper._kv_layout

    def _operator_before(self, ticket, args):
        self._check_ticket(ticket)
        if ticket.entered:
            raise RuntimeError('Repeated or reentrant selected low-level operation')
        ticket.entered = True
        p, layer, start, end = self.provider, ticket.layer, ticket.start, ticket.end
        _, _, attn, impl = p.layers[layer]
        cache = ticket.arguments['kv_cache']
        if ticket.wrapper is not None:
            layout = self._wrapper_plan(ticket, args)
            q, kv, out = args['q'], args['paged_kv_cache'], args['out']
        else:
            q, kv, out = args['query'], args['kv_cache'], args['out']
            layout = args['kv_layout']
            meta = ticket.arguments['attn_metadata'].decode
            if (layer != 0 or layout not in ('HND','NHD') or args['window_left'] != 1023
                    or args['q_len_per_req'] != 1 or args['max_seq_len'] != end
                    or args['block_tables'] is not meta.block_tables or args['seq_lens'] is not meta.seq_lens
                    or any(args[k] is not None for k in ('sinks','mask','q_cu_seq_lens','kv_cache_sf'))
                    or any(type(args[k]) not in (int,float) or args[k] != 1.0 for k in ('bmm1_scale','bmm2_scale','o_scale'))):
                raise RuntimeError('Actual XQA masks/scales/length/owner unsupported')
            ticket.binding.update(layout=layout, workspace=tensor_identity(args['workspace_buffer']),
                                  split_plan={'max_seq_len':end,'q_len_per_req':1,'window_left':1023})
        dim, heads = spec.LAYERS[layer]['dim'], spec.LAYERS[layer]['kv_heads']
        self._tensor(q, (end-start,16,dim)); self._tensor(out, (end-start,16,dim))
        self._same_region(q, ticket.arguments['query']); self._same_region(out, ticket.arguments['output'])
        if type(kv) is not tuple or len(kv) != 2:
            raise RuntimeError('Actual consumed K/V must be the two source-created cache views')
        order = tuple(impl.kv_cache_layout.layer_view_order)
        expected_order = (0,1,2,3) if layout == 'HND' else (0,2,1,3)
        if order != expected_order:
            raise RuntimeError('Wrapper layout differs from actual backend cache permutation')
        expected_shape = tuple(cache.shape[i] for i in order[:-1]) + (dim,)
        expected_stride = tuple(cache.stride()[i] for i in order)
        for i, view in enumerate(kv):
            ident = self._tensor(view, expected_shape)
            if (ident[1:3] != ticket.cache_identity[1:3]
                    or view.storage_offset() != cache.storage_offset()+i*dim
                    or tuple(view.stride()) != expected_stride or view.device != cache.device):
                raise RuntimeError('Consumed cache view does not bind exact owned K/V addresses')
        group = p.access.groups[p.layers[layer][0]][0]
        if (p.manager.sizes[group],p.manager.kernels[group],p.manager.ratios[group]) != (16,16,1):
            raise RuntimeError('Attention manager/kernel geometry changed')
        pages = (end+15)//16
        ids = list(p.manager.expanded(group)[:pages])
        tensors = []
        if ticket.wrapper is not None:
            w = ticket.wrapper
            tensors += [(w._paged_kv_indptr_buf, self._metadata(w._paged_kv_indptr_buf,(2,),[0,pages])),
                        (w._paged_kv_indices_buf, self._metadata(w._paged_kv_indices_buf,(pages,),ids)),
                        (w._paged_kv_last_page_len_buf,self._metadata(w._paged_kv_last_page_len_buf,(1,),[(end-1)%16+1]))]
            if ticket.entrypoint == 'paged_prefill' or w.use_tensor_cores:
                tensors.append((w._qo_indptr_buf,self._metadata(w._qo_indptr_buf,(2,),[0,end-start])))
        else:
            table = args['block_tables']
            if len(table.shape) != 2 or table.shape[0] != 1 or not pages <= table.shape[1] <= 144:
                raise RuntimeError('XQA table exceeds bounded owned request capacity')
            # Unconsumed tail slots are not required to be initialized.
            tensors += [(table, tensor_identity(table)), (args['seq_lens'],self._metadata(args['seq_lens'],(1,),[end]))]
            self._metadata(table[:,:pages],(1,pages),[ids])
        ticket.tensors = [(q,tensor_identity(q)), (out,tensor_identity(out)),
                          *((v,tensor_identity(v)) for v in kv), *tensors]
        ticket.binding.update(q=tensor_identity(q), out=tensor_identity(out),
                              cache_views=[tensor_identity(v) for v in kv],
                              tables=[identity for _,identity in tensors], thread=ticket.thread,
                              enclosing_impl=id(impl), enclosing_frame=id(ticket.frame))
        ticket.policy=self._policy(ticket)
        ticket.binding['policy']=ticket.policy
        tag=struct.pack('<III',layer,start,end)
        cache_hash=hashlib.sha256(b'attention-cache-inputs\0'+tag)
        positions=range(max(0,start-1023) if layer==0 else 0,end)
        slots=p.manager.slots(group,positions)
        # Reconstitute only the borrowed BHNC *view* from the verified actual
        # consumed K view. stored_row then independently indexes actual storage.
        consumed_cache=self.torch.as_strided(kv[0],tuple(cache.shape),tuple(cache.stride()),
                                            storage_offset=cache.storage_offset())
        with p.scratch.scope('attention_cache_entry'):
            for position in positions:
                remaining(p.deadline)
                key,value=stored_row(consumed_cache,slots[position],heads,dim,self.torch,p.transfer,'attention_cache_entry')
                selected=spec.select_writer_row(layer,position,key,value)
                pieces=tuple(data for _,_,data in selected)
                expected=self.writer_rows[layer].get(position)
                if (expected is None or expected != (slots[position],*(hashlib.sha256(data).digest() for data in pieces))
                        or pieces != p.evidence.read_selection(layer,position)):
                    raise RuntimeError('Actual consumed cache differs from processed writer archive')
                cache_hash.update(struct.pack('<I',position)); cache_hash.update(pieces[0]); cache_hash.update(pieces[1])
        request_hash=spec.digest(native_request_identity(p.plan,ticket.frame.request_id))
        inputs=ticket.frame.input_ids
        if (not isinstance(inputs,self.torch.Tensor) or inputs.dtype != self.torch.int32
                or tuple(inputs.shape)!=(end-start,)):
            raise RuntimeError('Actual attention frame input dtype/extent changed')
        if inputs.device.type=='cuda':
            size=inputs.numel()*4
            if self.metadata_bytes+size>65536:
                raise RuntimeError('Frame input observation exceeds precharged metadata allowance')
            self.metadata_bytes+=size
        inputs_hash=spec.digest(inputs.tolist())
        ticket.record={'layer':layer,'start':start,'end':end,'frame_input_ids_sha256':inputs_hash,
            'frame_identity_sha256':frame_identity(p.plan['plan_sha256'],request_hash,start,end,inputs_hash),
            'operator_binding_sha256':spec.digest(ticket.binding),
            'cache_mapping_sha256':spec.digest({'group':group,'manager':p.manager.receipt(),
                                               'actual_pages':ids,'slots':list(slots.items())}),
            'semantics':semantic_contract(layer),'dispatch':{
                'provider':'xqa' if ticket.entrypoint=='xqa_decode' else 'fa2',
                'entrypoint':ticket.entrypoint,'backend_source_sha256':BACKEND_SHA,
                'wrapper_source_sha256':WRAPPER_SHA['fa2' if ticket.entrypoint=='paged_prefill' else 'xqa'],
                'split_plan_sha256':spec.digest(ticket.binding['split_plan']),
                'kernel_binary_sha256':None,'kernel_profile_sha256':None,'native_rounding_contract':None},
            'cache_inputs_sha256':cache_hash.hexdigest(),
            'query_sha256':self._heads(ticket,q,'q')}
        return ticket

    def _heads(self,ticket,tensor,role):
        layer,start,end=ticket.layer,ticket.start,ticket.end
        result=hashlib.sha256(('attention-'+role+'\0').encode()+struct.pack('<III',layer,start,end))
        with self.provider.scratch.scope('attention_q_output'):
            for position in spec.POSITIONS:
                if start <= position < end:
                    result.update(struct.pack('<I',position))
                    for head in spec.LAYERS[layer]['q_heads']:
                        remaining(self.provider.deadline)
                        raw=native_bytes(tensor[position-start,head],self.torch,self.provider.transfer,'attention_q_output')
                        name,offset,data=spec.select_head_row(layer,position,head,role,raw)
                        self.provider.evidence.head_row(layer,position,head,role,raw)
                        result.update(data)
        return result.hexdigest()

    def _operator_after(self,ticket,result):
        self._check_ticket(ticket)
        if ticket.complete or not ticket.entered:
            raise RuntimeError('Repeated/missing selected operator completion')
        out=ticket.tensors[1][0]
        if result is not out:
            raise RuntimeError('Native operator did not return the exact passed output')
        for tensor,identity in ticket.tensors:
            if tensor_identity(tensor) != identity:
                raise RuntimeError('Actual operator tensor owner/view changed during call')
        if self._policy(ticket)!=ticket.policy:
            raise RuntimeError('Actual attention semantic/dispatch policy changed during call')
        for tensor,shape,expected in ticket.metadata_reads:
            self._metadata(tensor,shape,expected,record=False)
        if ticket.wrapper is not None:
            w=ticket.wrapper
            if (id(w._cached_module) != ticket.binding['kernel']
                    or id(getattr(w._cached_module,ticket.binding['kernel_name'])) != ticket.binding['kernel_callback']
                    or list(w._plan_info) != ticket.binding['split_plan']
                    or id(w._plan_info) != ticket.binding['split_plan_owner']
                    or _closure(getattr(w._cached_module,ticket.binding['kernel_name'])).get('paged_run_func') is not ticket.native_callback
                    or self.guard.native_leaf(ticket.native_callback) != ticket.binding['native_origin']):
                raise RuntimeError('Actual native plan/kernel owner changed during operation')
        else:
            if not ticket.kernel_seen or self.guard.native_leaf(ticket.native_callback) != ticket.binding['native_origin']:
                raise RuntimeError('Missing/changed resolved XQA native kernel owner')
            alias=ticket.hook.original
            if id(self._cached_xqa(alias)) != ticket.binding['lazy_inner']:
                raise RuntimeError('XQA cached implementation changed during call')
        # Timing is explicitly unqualified. Synchronize the producer globally,
        # including foreign producer streams, before any O host read/o_proj.
        self.torch.cuda.synchronize()
        ticket.record['operator_binding_sha256']=spec.digest(ticket.binding)
        ticket.record['output_sha256']=self._heads(ticket,out,'o')
        ticket.complete=True

    def _end(self,ticket,result):
        if ticket is None:
            return
        self._check_ticket(ticket)
        if not ticket.entered or not ticket.complete:
            raise RuntimeError('Enclosing attention call omitted selected native operation')
        if self._policy(ticket)!=ticket.policy:
            raise RuntimeError('Enclosing attention semantic lifetime changed')
        self._same_region(result,ticket.arguments['output'])
        expected=[(s,e,l) for s,e in spec.FRAMES for l in (0,5)]
        identity=(ticket.start,ticket.end,ticket.layer)
        if len(self.records)>=len(expected) or identity != expected[len(self.records)]:
            raise RuntimeError('Selected attention record order changed')
        self.records.append(ticket.record)
        self.seen.add(identity)

    def _final(self,ticket,primary):
        # A failed _begin may already have published an active ticket.
        selected=ticket or self.active_ticket
        if self.drain_failed:
            return  # Failed producer drain retains ownership for owned teardown.
        if selected is not None:
            if selected.hook is not None:
                selected.hook.restore()
            if selected.resolver_hook is not None:
                selected.resolver_hook.restore()
            self.active_ticket=None

    def require_current(self):
        if self.failed or self.restored:
            raise RuntimeError('Attention hook set is poisoned or restored')
        self.guard.require_current()
        for hook in self.hooks:
            hook.require_current()
        if self.active_ticket is not None and self.active_ticket.hook is not None:
            self.active_ticket.hook.require_current()

    def require_frame_complete(self,frame):
        self.require_current()
        if frame is not self.provider.frame or self.active_ticket is not None:
            raise RuntimeError('Attention frame completed outside exact owned ticket')
        start,end,_=self.provider.ledger.pending
        if (start,end) in spec.FRAMES and any((start,end,l) not in self.seen for l in (0,5)):
            raise RuntimeError('Frame omitted selected layer attention')

    def require_complete(self):
        if self.failed or self.active_ticket is not None or len(self.records)!=12:
            raise RuntimeError('Incomplete/poisoned attention native records')
        if any(set(rows)!=set(range(2049)) for rows in self.writer_rows.values()):
            raise RuntimeError('Incomplete original writer archive')
        if (self.provider.transfer.kinds.get('attention_cache_entry') != 73388032
                or self.provider.transfer.kinds.get('attention_q_output') != 102400
                or self.provider.transfer.kinds.get('attention_metadata') != 65536):
            raise RuntimeError('Exact additional attention transfer domains differ')
        return list(self.records)

    def restore(self,primary=None):
        if self.restored:
            return
        errors=[]
        hooks=([h for h in (self.active_ticket.hook,self.active_ticket.resolver_hook) if h is not None]
               if self.active_ticket is not None else [])+list(reversed(self.hooks))
        if any(h.active for h in hooks):
            try:
                self.torch.cuda.synchronize()
            except BaseException as error:
                self.failed=True
                self.drain_failed=True
                if primary is None:
                    raise RuntimeError('Attention producer drain failed; hooks retained for teardown') from error
                add_failure_note(primary,'Attention producer drain failed; hooks retained for teardown: '+repr(error))
                return
        for hook in hooks:
            try:
                hook.restore()
            except BaseException as error:
                errors.append(error)
        if errors:
            self.failed=True
            if primary is None:
                raise RuntimeError('Attention hook restoration failed') from errors[0]
            add_failure_note(primary,'Attention hook restoration failed: '+repr(errors[0]))
            return
        self.active_ticket=None
        self.restored=True
