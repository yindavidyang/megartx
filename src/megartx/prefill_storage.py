"""CPU-only mechanics for the default-off native storage control.

Expected addresses originate in manager append inputs, never writer slots or
observer mapping helpers. Tensor functions accept an already-owned runtime;
importing this module loads neither Torch nor vLLM.
"""
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
import struct
from types import MethodType


class TransferBudget:
    """Charge before D2H; inherited metadata uses a source-audited upper bound."""
    def __init__(self, limit=4 << 30):
        if type(limit) is not int or not 0 < limit <= 4 << 30:
            raise ValueError('Invalid storage transfer limit')
        self.limit, self.bytes, self.calls = limit, 0, 0
        self.kinds = {}

    def reserve(self, size, kind):
        if type(size) is not int or size < 0 or self.bytes + size > self.limit:
            raise RuntimeError('Storage D2H cap rejected before transfer')
        self.bytes += size
        self.calls += 1
        self.kinds[kind] = self.kinds.get(kind, 0) + size

    def receipt(self):
        return {'limit_bytes': self.limit, 'transferred_bytes': self.bytes,
                'copy_calls': self.calls, 'domains': dict(self.kinds)}


def exact_int(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Expected an exact nonnegative integer')
    return value


class ManagerInputs:
    """Snapshot actual append_block_ids BEFORE the selected native subdivision.

    No reverse reconstruction from a kernel table is permitted. Even overwrite
    calls must preserve the previously observed manager prefix for this one
    full-context request. Tables and append argument lists are copied to tuples.
    """
    def __init__(self, manager_sizes, kernel_sizes, ratios):
        self.sizes = tuple(manager_sizes)
        self.kernels = tuple(kernel_sizes)
        self.ratios = tuple(ratios)
        if not self.sizes or len(self.sizes) != len(self.kernels) or len(self.sizes) != len(self.ratios):
            raise ValueError('Actual manager/kernel group geometry required')
        for b, k, r in zip(self.sizes, self.kernels, self.ratios):
            if exact_int(k, 1) != 16 or exact_int(b, 1) != k * exact_int(r, 1):
                raise ValueError('Only actual16-token kernels with source-derived subdivision admitted')
        self.req_index, self.tables, self.calls, self.poisoned = None, None, 0, False

    def append(self, req_index, new_block_ids, overwrite):
        try:
            if self.poisoned or type(overwrite) is not bool:
                raise ValueError('Poisoned manager receipt or nonboolean overwrite')
            exact_int(req_index)
            if self.req_index is not None and req_index != self.req_index:
                raise ValueError('More than one manager request row')
            if (type(new_block_ids) is not tuple or len(new_block_ids) != len(self.sizes)
                    or any(type(x) is not list for x in new_block_ids)):
                raise ValueError('Actual manager append tuple/list contract changed')
            if self.tables is None and overwrite is not True:
                raise ValueError('Missing first actual manager overwrite')
            old = self.tables or tuple(() for _ in self.sizes)
            pending = []
            for group, values in enumerate(new_block_ids):
                values = tuple(exact_int(v, 1) for v in values)
                current = values if overwrite else old[group] + values
                if (len(set(current)) != len(current) or current[:len(old[group])] != old[group]
                        or len(current) > (2304 + self.sizes[group]-1)//self.sizes[group]):
                    raise ValueError('Recycled/reordered/oversized full-context manager allocation')
                pending.append(current)
            self.tables, self.req_index = tuple(pending), req_index
            self.calls += 1
        except BaseException:
            self.poisoned = True
            raise

    def expanded(self, group):
        if self.poisoned or self.tables is None:
            raise ValueError('No unpoisoned original manager inputs')
        exact_int(group)
        if group >= len(self.tables):
            raise ValueError('Unknown cache group')
        ratio = self.ratios[group]
        return tuple(manager * ratio + child for manager in self.tables[group]
                     for child in range(ratio))

    def slots(self, group, positions):
        expanded = self.expanded(group)
        b, k, ratio = self.sizes[group], self.kernels[group], self.ratios[group]
        result = {}
        for position in positions:
            exact_int(position)
            manager_index, within = divmod(position, b)
            if manager_index >= len(self.tables[group]):
                raise ValueError('Original manager table misses required prefix/tail')
            manager = self.tables[group][manager_index]
            kernel = manager * ratio + within // k
            # Separate arithmetic on actual manager IDs; expanded is only a
            # sanity check here, never the source of expected addresses.
            if expanded[position // k] != kernel:
                raise ValueError('Manager subdivision inconsistency')
            result[position] = kernel * k + within % k
        return result

    def reconcile(self, group, persistent, gathered, count, end):
        expected = self.expanded(group)
        if (type(count) is not int or count != len(expected)
                or tuple(persistent) != expected
                or tuple(gathered) != expected[:(end+self.kernels[group]-1)//self.kernels[group]]):
            self.poisoned = True
            raise ValueError('Actual persistent/gathered kernel table differs from original manager inputs')

    def receipt(self):
        if self.tables is None:
            raise ValueError('Manager allocation was not observed')
        return {'manager_block_tokens': list(self.sizes), 'kernel_block_tokens': list(self.kernels),
                'manager_to_kernel_ratios': list(self.ratios), 'append_calls': self.calls,
                'manager_table_sha256': hashlib.sha256(json.dumps(self.tables, separators=(',', ':')).encode()).hexdigest(),
                'expected_address_source': 'actual_append_block_ids_before_native_subdivision'}


class InstanceHook:
    """Own one exact selected bound callback and restore only our own wrapper."""
    def __init__(self, owner, name, before, after, fail):
        if name in vars(owner):
            raise RuntimeError('Unexpected preexisting instance callback override: ' + name)
        original = getattr(owner, name)
        if type(original) is not MethodType or original.__self__ is not owner:
            raise RuntimeError('Expected actual native bound method: ' + name)
        self.owner, self.name, self.original = owner, name, original
        self.class_callback, self.active = getattr(type(owner), name), True
        self.signature = inspect.signature(original)
        self.fail = fail
        @wraps(original)
        def wrapped(actual, *args, **kwargs):
            try:
                self.require_current()
                if actual is not owner:
                    raise RuntimeError('Selected callback owner changed')
                bound = self.signature.bind(*args, **kwargs)
                bound.apply_defaults()
                ticket = before(bound.arguments)
                result = original(*args, **kwargs)
                after(ticket, result)
                return result
            except BaseException as error:
                fail(error)
                raise
        self.wrapper = wrapped
        setattr(owner, name, MethodType(wrapped, owner))

    def require_current(self):
        callback = getattr(self.owner, self.name, None)
        if (not self.active or getattr(type(self.owner), self.name) is not self.class_callback
                or type(callback) is not MethodType or callback.__self__ is not self.owner
                or callback.__func__ is not self.wrapper):
            raise RuntimeError('Native callback class/instance replacement: ' + self.name)

    def restore(self):
        if not self.active:
            return
        self.require_current()
        delattr(self.owner, self.name)
        self.active = False
        restored = getattr(self.owner, self.name)
        if restored.__self__ is not self.owner or restored.__func__ is not self.original.__func__:
            raise RuntimeError('Native callback restoration changed owner')


class SpecialCallHook:
    """Python resolves __call__ on the class; bind one existing sampler only."""
    def __init__(self, owner, before, after, fail):
        if '__call__' in vars(owner):
            raise RuntimeError('Sampler instance shadows selected call')
        self.owner, self.cls, self.active = owner, type(owner), True
        self.original = self.cls.__call__
        self.signature = inspect.signature(self.original)
        @wraps(self.original)
        def wrapped(actual, *args, **kwargs):
            if actual is not owner:
                return self.original(actual, *args, **kwargs)
            try:
                self.require_current()
                bound = self.signature.bind(actual, *args, **kwargs)
                bound.apply_defaults()
                ticket = before(bound.arguments)
                result = self.original(actual, *args, **kwargs)
                after(ticket, result)
                return result
            except BaseException as error:
                fail(error)
                raise
        self.wrapper = wrapped
        self.cls.__call__ = wrapped

    def require_current(self):
        if (not self.active or type(self.owner) is not self.cls
                or self.cls.__call__ is not self.wrapper or '__call__' in vars(self.owner)):
            raise RuntimeError('Actual sampler call ownership changed')

    def restore(self):
        if not self.active:
            return
        self.require_current()
        self.cls.__call__ = self.original
        self.active = False


def require_source(owner, module, name, expected_hash):
    """Check loaded class and executing file, not merely an adjacent snapshot."""
    if type(owner).__module__ != module or type(owner).__name__ != name:
        raise RuntimeError('Unexpected actual source owner: ' + module)
    path = Path(inspect.getsourcefile(type(owner)))
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise RuntimeError('Actual native callback source drift: ' + module)


def native_bytes(tensor, torch, budget, kind):
    """Synchronous original-bit D2H; never contiguous/cast on the GPU."""
    if tensor.dtype != torch.bfloat16 or tensor.device.type != 'cuda':
        raise RuntimeError('Original CUDA BF16 bytes required')
    budget.reserve(tensor.numel() * 2, kind)
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def stored_row(cache, slot, heads, dim, torch, budget, kind):
    """Independent physical storage-offset/stride read, at most one K/V row.

    Does not import gather_writer_rows or any prior observer hashing/mapping
    helper. as_strided creates a borrowed view only; .cpu precedes contiguous.
    """
    if (type(slot) is not int or slot < 0 or len(cache.shape) != 4
            or tuple(cache.shape[1:]) != (heads, 16, 2*dim)):
        raise RuntimeError('Actual BF16 BHNC storage geometry differs')
    block, token = divmod(slot, 16)
    if block >= cache.shape[0]:
        raise RuntimeError('Independent physical slot outside owned cache')
    strides = cache.stride()
    base = cache.storage_offset() + block*strides[0] + token*strides[2]
    extent = base + (heads-1)*strides[1] + (2*dim-1)*strides[3] + 1
    if min(strides) <= 0 or strides[3] != 1 or base < 0 or extent*2 > cache.untyped_storage().nbytes():
        raise RuntimeError('Independent physical row exceeds storage')
    key = torch.as_strided(cache, (heads, dim), (strides[1], strides[3]), storage_offset=base)
    value = torch.as_strided(cache, (heads, dim), (strides[1], strides[3]), storage_offset=base+dim)
    return native_bytes(key, torch, budget, kind), native_bytes(value, torch, budget, kind)


def canonical_runtime(value):
    """No repr/object-address fallback for actual SamplingParams identity."""
    from enum import Enum
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if value != value or value in (float('inf'), float('-inf')):
            raise ValueError('Nonfinite runtime identity')
        return value
    if isinstance(value, Enum):
        return {'enum': type(value).__module__ + '.' + type(value).__qualname__,
                'value': canonical_runtime(value.value)}
    if type(value) in (list, tuple):
        return [canonical_runtime(x) for x in value]
    if type(value) in (set, frozenset):
        return sorted((canonical_runtime(x) for x in value), key=lambda x: json.dumps(x, sort_keys=True))
    if type(value) is dict:
        if any(type(k) is not str for k in value):
            raise ValueError('Runtime identity mapping requires unambiguous string keys')
        return {k: canonical_runtime(v) for k, v in value.items()}
    fields = getattr(type(value), '__struct_fields__', None)
    if fields is not None:
        return {name: canonical_runtime(getattr(value, name)) for name in fields}
    raise ValueError('Unsupported actual runtime identity type: ' + type(value).__name__)


def table_owner_signature(table):
    """Freeze identities of real manager table objects, not just equal values."""
    def tensor(t):
        return (id(t), t.data_ptr(), tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device))
    def numpy_alias(array, cpu):
        interface = array.__array_interface__
        if (interface['data'][0] != cpu.data_ptr() or tuple(array.shape) != tuple(cpu.shape)
                or tuple(s // array.itemsize for s in array.strides) != tuple(cpu.stride())):
            raise RuntimeError('Manager NumPy source no longer aliases its owned CPU tensor')
        return (id(array), interface['data'][0], tuple(array.shape), tuple(array.strides), str(array.dtype))
    source_numpy = numpy_alias(table.num_blocks.np, table.num_blocks.cpu)
    pool, active = table.num_blocks.pool, table.num_blocks.gpu
    if (type(pool._curr) is not int or not 0 <= pool._curr < len(pool._uva_bufs)
            or len(pool._uva_bufs) != pool.max_concurrency):
        raise RuntimeError('Invalid actual manager UVA pool cursor')
    selected = pool._uva_bufs[pool._curr]._uva
    if (active.data_ptr() != selected.data_ptr() or tuple(active.shape) != tuple(table.num_blocks.cpu.shape)
            or tuple(active.stride()) != tuple(selected.stride()) or active.dtype != selected.dtype
            or active.device != selected.device):
        raise RuntimeError('Manager active UVA view escaped its actual unchanged pool')
    return (id(table), id(table.block_sizes), id(table.kernel_block_sizes),
            id(table.blocks_per_kv_block), id(table.block_tables),
            tuple((id(item), tensor(item.gpu)) for item in table.block_tables),
            id(table.input_block_tables), tuple(tensor(t) for t in table.input_block_tables),
            id(table.num_blocks), source_numpy, tensor(table.num_blocks.cpu),
            id(table.num_blocks.pool), id(table.num_blocks.pool._uva_bufs),
            tuple((id(buf), numpy_alias(buf.np, buf.cpu), tensor(buf.cpu), tensor(buf._uva)) for buf in table.num_blocks.pool._uva_bufs))


def require_method_source(owner, name):
    """Reject pre-install callback substitution even when class files match."""
    callback = getattr(owner, name)
    if (type(callback) is not MethodType or callback.__self__ is not owner
            or callback.__func__ is not getattr(type(owner), name)
            or callback.__func__.__qualname__ != type(owner).__name__+'.'+name
            or callback.__func__.__code__.co_name != name
            or inspect.getsourcefile(callback.__func__) != inspect.getsourcefile(type(owner))):
        raise RuntimeError('Selected native method is not its source-defined owner: '+name)
