"""
If we run this code in the scico environment, it will raise an error.
The command line must exactly be CUDA_VISIBLE_DEVICES=0,1,2,3 python -u bug_test.py in order to reproduce the error.
I have no idea why this is happening. Perhaps worth investigating.
"""

# import mbirjax
# # Maybe mbirjax is calling a different config?
import jax
# FIX: Maybe call jax.config?


if __name__ == "__main__":
    print(jax.devices('gpu'))
    # for i in range(31, 500):
        # print(f"Iteration {i}")
    test_arr = jax.numpy.zeros((128, 30, 128))
    test_arr = jax.device_put(test_arr, jax.devices('gpu')[0])
    import mbirjax
    test_arr.transpose(1, 0, 2)
