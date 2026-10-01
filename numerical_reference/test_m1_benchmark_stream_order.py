"""CPU source guard for the reviewed nonblocking-stream dependency; no GPU proof."""
from pathlib import Path
import re
import unittest


def require_owned_stream(source):
    source = re.sub(r'//[^\n]*', '', source)
    buffer = source.split('struct Buffer {', 1)[1].split('struct Inputs {', 1)[0]
    inputs = source.split('struct Inputs {', 1)[1].split('struct Outputs {', 1)[0]
    if re.search(r'\bcuda(?:Memcpy|Memset)\s*\(', source):
        raise AssertionError('default-stream transfer/initialization reintroduced')
    calls = re.findall(r'cuda(?:MemcpyAsync|MemsetAsync)\((.*?)\)\s*\)', buffer, re.S)
    if len(calls) != 5 or any(call.rsplit(',', 1)[-1].strip() != 'stream' for call in calls):
        raise AssertionError('every buffer operation must use its owned stream')
    if 'cudaStream_t const stream;' not in buffer:
        raise AssertionError('buffer must retain the supplied stream')
    if 'Inputs input{stream};' not in source or 'Outputs fused{stream}, control{stream};' not in source:
        raise AssertionError('buffers must share the kernel stream')
    upload = inputs.split('void upload(Bytes const& raw)', 1)[1].split('Bytes snapshot()', 1)[0]
    if upload.count('.upload(') != 4 or 'check(cudaStreamSynchronize(ids.stream));' not in upload:
        raise AssertionError('pageable input lifetime requires checked upload completion')
    if upload.index('sf.upload(') > upload.index('check(cudaStreamSynchronize(ids.stream));'):
        raise AssertionError('the final SF upload must precede the completion fence')
    snapshot = buffer.split('Bytes snapshot(', 1)[1]
    if 'check(cudaStreamSynchronize(stream));' not in snapshot:
        raise AssertionError('download must complete before host inspection')
    if snapshot.index('check(cudaStreamSynchronize(stream));') > snapshot.index('std::all_of'):
        raise AssertionError('host inspection must follow download completion')


class M1BenchmarkStreamOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (Path(__file__).resolve().parents[1] / 'benchmarks/m1_maps_expand.cu').read_text()

    def test_setup_transfers_and_consumers_share_the_owned_stream(self):
        require_owned_stream(self.source)

    def test_known_missing_dependency_and_host_lifetime_mutations_fail(self):
        mutations = (
            self.source.replace('cudaMemcpyAsync(data(), bytes, size, cudaMemcpyHostToDevice, stream)',
                                'cudaMemcpy(data(), bytes, size, cudaMemcpyHostToDevice)'),
            self.source.replace('cudaMemsetAsync(storage, 0xA7, 32, stream)',
                                'cudaMemsetAsync(storage, 0xA7, 32, nullptr)'),
            self.source.replace('check(cudaStreamSynchronize(ids.stream));', ''),
            self.source.replace('check(cudaStreamSynchronize(stream));  // Finish D2H', '// Finish D2H'),
            self.source.replace('Inputs input{stream};', 'Inputs input{nullptr};'),
        )
        for source in mutations:
            with self.subTest(mutation=mutations.index(source)), self.assertRaises(AssertionError):
                require_owned_stream(source)


if __name__ == '__main__':
    unittest.main()
