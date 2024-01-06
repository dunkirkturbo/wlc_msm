## Configuration

Our implementation based on cached Jacobian coordinates is evaluated on three different graphics cards: 1) V100, 2) RTX3090, and 3) RTX4090. Detailed hardware information including the CPU configuration on the host side is listed below.

| Environment  | V100                   | RTX3090                | RTX4090                |
| ------------ | ---------------------- | ---------------------- | ---------------------- |
| Device       | V100-SXM2-32GB         | GeForce RTX3090        | GeForce RTX4090        |
| SM Count     | 80                     | 82                     | 128                    |
| Core Count   | 5120                   | 10496                  | 16384                  |
| Host(CPU)    | Xeon(R) Platinum 8255C | Xeon(R) Platinum 8358P | Xeon(R) Platinum 8352V |
| CPU Cores    | 12                     | 15                     | 12                     |
| CPU Freq.    | 2.50GHz                | 2.60GHz                | 2.10GHz                |
| OS           | Ubuntu 20.04           | Ubuntu 20.04           | Ubuntu 20.04           |
| CUDA Version | 11.3                   | 11.3                   | 11.8                   |

Our GPU Implemetation of MSM relies on [`CUB`](https://nvlabs.github.io/cub/), which provides state-of-the-art, reusable software components for every layer of the CUDA programming model. By default, CUB is included in the CUDA Toolkit.

### Install Rust

Install the Rust toolchain by:

```shell
sudo curl https://sh.rustup.rs -sSf | sh
source ~/.cargo/env
```

### Optional: Build and run cuZK that we compare with

```shell
apt-get update
apt-get upgrade
apt-get install libgmp-dev	# which cuZK required
git clone https://github.com/speakspeak/cuZK.git
cd cuZK/test
make msmb
./ msmtestb 20	# run a test of an MSM of 2^20 scale on the BLS12-381 curve
```

## Support

If you don't have the proper device around to evaluate our implementation, you can go to website `AutoDL`(https://www.autodl.com/) for gpu server rental, which is exactly our experimental environment.

## Build

### 381_xyzz_bal

To evaluate our implementation **based on cached Jacobian coordinates**:

```shell
cd 381_xyzz_bal
cargo build --release
```

To benchmark our MSM implementation, run:

```shell
cargo bench
```

To test the correctness of our MSM implementation, run:

```shell
cargo test --release
```

### 381_xyzz_constant

The parameter $\tau$ is set as $15$ to accumulate parts of the buckets into buffers in constant-time. And the new algorithm for aggregating the buffered points into buckets is shown in Section 5.

If you want to modify the sampling of input scalars, you can make changes in the `generate_points_scalars` function in `src/util.rs`.

To benchmark our MSM implementation on the BLS12-381 curve, run:

```shell
cd 381_xyzz_constant
cargo build --release
cargo bench
```

Then the results should be divided by the parameter `batches`.

## Main Functions

* 381_xyzz_bal/sppark/msm/pippenger.cuh
  * kernel function `process_scalar_1`: conversion of the sub-scalars (table lookups).
  * kernel function `bucket_acc`: accumulate parts of the buckets into static buffers.
  * kernel function `bucket_acc_2`: aggregate the buffered points into the buckets.
  * kernel function `bucket_agg_1` and `bucket_agg_2`: our parallel layered reduction algorithm for the bucket aggregation phase.
  * kernel function `recursive_sum`: final summation in each Pippenger subtask.