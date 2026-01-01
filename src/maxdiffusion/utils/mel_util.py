import jax.numpy as jnp
import jax.scipy as jsp
import librosa


import jax

def dynamic_range_compression_jax(x, C=1, clip_val=1e-7):
    return jnp.log(jnp.clip(x,min=clip_val) * C)

@jax.jit(static_argnames=['n_mels', 'n_fft', 'win_size', 'hop_length', 'fmin', 'fmax', 'sampling_rate'])
def get_mel(y, n_mels=100,n_fft=1024,win_size=1024,hop_length=256,fmin=0,fmax=None,clip_val=1e-7,sampling_rate=24000):
    # Librosa style mel filterbank
    mel_filters = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_filters = jnp.array(mel_filters)

    if y.ndim == 1:
        y = y[jnp.newaxis, :]

    # Padding to match center=True
    pad_amount = n_fft // 2
    y_padded = jnp.pad(y, ((0, 0), (pad_amount, pad_amount)), mode='reflect')

    window = jnp.hanning(win_size)
    
    # jax.scipy.signal.stft
    f, t, Zxx = jsp.signal.stft(y_padded, fs=sampling_rate, window=window, nperseg=win_size, noverlap=win_size-hop_length, nfft=n_fft, boundary=None, return_onesided=True, padded=False, axis=-1)
    
    mag = jnp.abs(Zxx) # (Batch, Freq, Time)
    
    # mel_filters: (Mels, Freq)
    # mag: (Batch, Freq, Time)
    # tensordot(axes=([1], [1])) -> (Batch, Time, Mels)
    mel_spec = jnp.tensordot(mag, mel_filters, axes=([1], [1])) 
    
    # Transpose to (Batch, Mels, Time) to match audax/pytorch convention
    mel_spec = jnp.swapaxes(mel_spec, -1, -2) 

    spec = dynamic_range_compression_jax(mel_spec, clip_val=clip_val)
    return spec