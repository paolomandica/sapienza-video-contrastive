# README for `utils`

Some light-touch documentation for a couple of things in `utils`. 

### Note on Parallelised Superpixels Implementation

In order to implement the calculation of superpixel representations in a parallel (vectorised) manner, i.e. to delegate work to PyTorch and therefore low-level optimised routines with GPU support, we had to implement the [`view_as_windows`](https://github.com/paolomandica/sapienza-video-contrastive/blob/7fe681ef1499c1340edc1f2acc5c52f0a55b39f2/code/utils/__init__.py#L433) function from `skimage` using PyTorch. 

A few cursory notes on the implementation:

- The [original `view_as_windows` function](https://github.com/scikit-image/scikit-image/blob/2dbd57757e7ad32cf2f5f3cdfc45996c55526d1a/skimage/util/shape.py#L97) that our reimplementation is based on can be found under [scikit-image/skimage/util/shape.py](https://github.com/scikit-image/scikit-image/blob/main/skimage/util/shape.py)
- This makes use of a number of generally non-user-level functions which access or mutate the views of the data that NumPy arrays or PyTorch tensors have, in particular:
  - NumPy's [as_strided](https://github.com/numpy/numpy/blob/b235f9e701e14ed6f6f6dcba885f7986a833743f/numpy/lib/stride_tricks.py#L38-L114) function, which _creates a view into the array with the given shape and strides_; found in [numpy/numpy/lib/stride_tricks.py](https://github.com/numpy/numpy/blob/v1.21.0/numpy/lib/stride_tricks.py). 
  - PyTorch's equivalent, [torch.as_strided](https://pytorch.org/docs/stable/generated/torch.as_strided.html)
  - The [Tensor.stride](https://pytorch.org/docs/stable/generated/torch.Tensor.stride.html) method, which _returns the stride of self tensor_ and NumPy's version [numpy.ndarray.strides](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.strides.html)
  - See also [_maybe_view_as_subclass](https://github.com/numpy/numpy/blob/b235f9e701e14ed6f6f6dcba885f7986a833743f/numpy/lib/stride_tricks.py#L25) from the same module

Note that some of the above are structured differently so Torch generally uses class methods to access attributes of Tensors whereas NumPy uses straight attributes at the API level. For example `numpy.ndarray.strides` versus `torch.Tensor.stride(dim)` or the more familiar [`torch.Tensor.size()`](https://pytorch.org/docs/stable/generated/torch.Tensor.size.html) (method; also [aliased](https://stackoverflow.com/a/63263351/6619692) as `torch.Tensor.shape`) and [`numpy.ndarray.shape`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html) (attribute)
