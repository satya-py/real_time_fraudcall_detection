import numpy as np

class RollingBuffer:
    def __init__(self, buffer_duration_sec, sample_rate):
        self.buffer_size = int(buffer_duration_sec * sample_rate)
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.ptr = 0
        self.filled = False

    def add_chunk(self, chunk):
        """
        Adds a new chunk of audio to the circular buffer.
        chunk: numpy array of float32 samples.
        """
        chunk_len = len(chunk)
        if chunk_len > self.buffer_size:
            # If chunk is larger than buffer, take the last buffer_size samples
            self.buffer[:] = chunk[-self.buffer_size:]
            self.ptr = 0
            self.filled = True
            return

        end_ptr = self.ptr + chunk_len
        
        if end_ptr <= self.buffer_size:
            self.buffer[self.ptr:end_ptr] = chunk
            self.ptr = end_ptr
            if self.ptr == self.buffer_size:
                 self.ptr = 0
                 self.filled = True
        else:
            # Wrap around
            overflow = end_ptr - self.buffer_size
            self.buffer[self.ptr:] = chunk[:-overflow]
            self.buffer[:overflow] = chunk[-overflow:]
            self.ptr = overflow
            self.filled = True

    def get_buffer(self):
        """
        Returns the current buffer content in correct temporal order.
        """
        if not self.filled:
            return self.buffer[:self.ptr]
        else:
            return np.concatenate((self.buffer[self.ptr:], self.buffer[:self.ptr]))

    def get_last_n_seconds(self, n_seconds, sample_rate):
        """
        Returns the last n seconds of audio.
        """
        num_samples = int(n_seconds * sample_rate)
        full_buffer = self.get_buffer()
        if len(full_buffer) < num_samples:
            return full_buffer
        return full_buffer[-num_samples:]
