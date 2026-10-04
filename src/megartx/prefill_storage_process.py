"""CPU-only, single-child client execution with a strict console byte budget.

The source-bound storage client must not create subprocesses. Cleanup therefore
owns only the direct ``Popen`` child; it never signals a process group or looks
up other PIDs. A receipt is attempted once, including on launch failures.
"""
import base64
import math
import os
import selectors
import subprocess
import time


MAX_CONSOLE_BYTES = 65536
MAX_READ_BYTES = 4096
TERM_GRACE_SECONDS = 0.25
KILL_GRACE_SECONDS = 1.0
CONSOLE_SCHEMA = 'megartx-prefill-storage-client-console-v1'


def _remaining(deadline):
    if (isinstance(deadline, bool) or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline)):
        raise ValueError('Storage client requires a finite monotonic deadline')
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Storage client deadline expired')
    return remaining


def _note(primary, context, error):
    """Keep cleanup failures subordinate, including on Python 3.10."""
    try:
        note = context + ': ' + type(error).__name__ + ': ' + str(error)[:512]
        if hasattr(primary, 'add_note'):
            primary.add_note(note)
        else:
            notes = list(getattr(primary, '__notes__', ()))
            notes.append(note)
            primary.__notes__ = notes
    except BaseException:
        # Even broken exception formatting must not replace the primary error.
        pass


def _stop_owned_child(process, primary):
    """Bound TERM/KILL reaping grace and act only through the owned Popen."""
    try:
        if process.poll() is not None:
            return  # poll has already reaped this direct child.
    except BaseException as error:
        _note(primary, 'Storage client cleanup poll failed', error)
    try:
        process.terminate()
    except BaseException as error:
        _note(primary, 'Storage client cleanup terminate failed', error)
    try:
        process.wait(timeout=TERM_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    except BaseException as error:
        _note(primary, 'Storage client cleanup TERM wait failed', error)
    try:
        process.kill()
    except BaseException as error:
        _note(primary, 'Storage client cleanup kill failed', error)
    try:
        process.wait(timeout=KILL_GRACE_SECONDS)
    except BaseException as error:
        _note(primary, 'Storage client cleanup KILL reap failed', error)


def _status(primary, overflow):
    if overflow:
        return 'overflow'
    if isinstance(primary, TimeoutError):
        return 'timeout'
    if isinstance(primary, InterruptedError):
        return 'interrupted'
    if primary is not None:
        return 'error' if isinstance(primary, Exception) else 'interrupted'
    return 'completed'


def run_bounded_client(command, env, deadline, evidence):
    """Run once, retaining at most 64 KiB of merged stdout/stderr bytes.

    ``deadline`` is an absolute ``time.monotonic()`` deadline. Every pipe read
    is at most 4096 bytes and at most the remaining budget plus one overflow
    sentinel. At exactly the cap, EOF is still accepted; another byte fails
    immediately, without waiting for the client to finish. Nonzero exits are
    returned to the caller, which owns their interpretation.

    ``evidence.write(name, value)`` receives one bounded, binary-safe console
    receipt. ``output_bytes`` counts retained bytes, not an unbounded estimate
    of bytes the child might have produced. Base64 is at most 87384 bytes.
    Evidence/cleanup failures never replace an earlier execution failure.
    """
    process = selector = None
    output = bytearray()
    primary = None
    overflow = False
    try:
        _remaining(deadline)
        process = subprocess.Popen(
            command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=0,
        )
        _remaining(deadline)
        os.set_blocking(process.stdout.fileno(), False)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        eof = False
        while not eof:
            events = selector.select(timeout=_remaining(deadline))
            _remaining(deadline)
            for key, _ in events:
                space = MAX_CONSOLE_BYTES - len(output)
                try:
                    chunk = os.read(key.fd, min(MAX_READ_BYTES, space + 1))
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    eof = True
                    break
                output.extend(chunk[:space])
                if len(chunk) > space:
                    overflow = True
                    raise ValueError('Storage client console exceeded 65536 bytes')
                _remaining(deadline)
        try:
            process.wait(timeout=_remaining(deadline))
        except subprocess.TimeoutExpired as error:
            raise TimeoutError('Storage client deadline expired') from error
        _remaining(deadline)
    except BaseException as error:
        primary = error
    finally:
        if primary is not None and process is not None:
            _stop_owned_child(process, primary)
        for resource in (selector, process.stdout if process is not None else None):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    _note(primary, 'Storage client resource close failed', error)
        if primary is None:
            try:
                _remaining(deadline)
            except BaseException as error:
                primary = error
        receipt = {
            'schema': CONSOLE_SCHEMA,
            'status': _status(primary, overflow),
            'returncode': process.returncode if process is not None else None,
            'output_bytes': len(output),
            'overflow': overflow,
            'console_base64': base64.b64encode(output).decode('ascii'),
        }
        try:
            evidence.write('client-console.json', receipt)
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                _note(primary, 'Storage client console evidence write failed', error)
    if primary is not None:
        raise primary
    _remaining(deadline)
    return subprocess.CompletedProcess(command, process.returncode, stdout=bytes(output))
