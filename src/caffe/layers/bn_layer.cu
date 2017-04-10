#include <algorithm>
#include <vector>

#include "caffe/layers/bn_layer.hpp"

namespace caffe {
  template <typename Dtype>
  void BNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* const_top_data = top[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    Dtype* spatial_mean_data = spatial_mean_.mutable_gpu_data();
    Dtype* buffer_data = buffer_blob_.mutable_gpu_data();
    const Dtype* const_buffer_data = buffer_blob_.gpu_data();

    switch (this->layer_param_.bn_param().bn_mode()) {
    case BNParameter_BNMode_LEARN:
      // put the squares of bottom into buffer_blob_
      caffe_gpu_powx(bottom[0]->count(), bottom_data, Dtype(2),
          buffer_blob_.mutable_gpu_data());

      // computes variance using var(X) = E(X^2) - (EX)^2
      // EX across spatial
      caffe_gpu_gemv<Dtype>(CblasNoTrans, N_ * C_, H_ * W_,
          Dtype(1. / (H_ * W_)),
          bottom_data, spatial_sum_multiplier_.gpu_data(),
          Dtype(0), spatial_mean_data);
      // EX across batch
      caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_, Dtype(1. / N_),
          spatial_mean_.gpu_data(),
          batch_sum_multiplier_.gpu_data(), Dtype(0),
          batch_mean_.mutable_gpu_data());

      // E(X^2) across spatial
      caffe_gpu_gemv<Dtype>(CblasNoTrans, N_ * C_, H_ * W_,
          Dtype(1. / (H_ * W_)), buffer_data,
          spatial_sum_multiplier_.gpu_data(), Dtype(0),
          spatial_variance_.mutable_gpu_data());
      // E(X^2) across batch
      caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_, Dtype(1. / N_),
          spatial_variance_.gpu_data(),
          batch_sum_multiplier_.gpu_data(), Dtype(0),
          batch_variance_.mutable_gpu_data());

      caffe_gpu_powx(batch_mean_.count(), batch_mean_.gpu_data(),
          Dtype(2), buffer_blob_.mutable_gpu_data());  // (EX)^2
      caffe_gpu_sub(batch_mean_.count(), batch_variance_.gpu_data(),
          buffer_data, batch_variance_.mutable_gpu_data());  // variance

      // save top[1] (batch_mean) and top[2] (batch_variance)
      if (top.size() > 1) {
          caffe_copy(batch_mean_.count(), batch_mean_.gpu_data(),
              top[1]->mutable_gpu_data());
      }
      if (top.size() > 2) {
          caffe_copy(batch_variance_.count(), batch_variance_.gpu_data(),
              top[2]->mutable_gpu_data());
      }

      // do mean and variance normalization
      // subtract mean
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
          batch_sum_multiplier_.gpu_data(), batch_mean_.gpu_data(), Dtype(0),
          spatial_mean_data);
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_, H_ * W_,
          1, -Dtype(1),
          spatial_mean_.gpu_data(), spatial_sum_multiplier_.gpu_data(), Dtype(0),
          buffer_blob_.mutable_gpu_data());

      caffe_gpu_add(buffer_blob_.count(), bottom_data, buffer_data, top_data);

      // normalize variance
      caffe_gpu_add_scalar(batch_variance_.count(), var_eps_,
          batch_variance_.mutable_gpu_data());
      caffe_gpu_powx(batch_variance_.count(), batch_variance_.gpu_data(),
          Dtype(0.5), batch_variance_.mutable_gpu_data());

      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
          batch_sum_multiplier_.gpu_data(), batch_variance_.gpu_data(), Dtype(0),
          spatial_variance_.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_,
          H_ * W_, 1, Dtype(1),
          spatial_variance_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
          Dtype(0), buffer_blob_.mutable_gpu_data());

      caffe_gpu_div(buffer_blob_.count(), top_data, buffer_data, top_data);

      // Saving x_norm
      caffe_copy(top[0]->count(), const_top_data, x_norm_.mutable_gpu_data());

      // scale
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
          batch_sum_multiplier_.gpu_data(), this->blobs_[0]->gpu_data(),
          Dtype(0), spatial_variance_.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_,
          H_ * W_, 1, Dtype(1),
          spatial_variance_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
          Dtype(0), buffer_blob_.mutable_gpu_data());

      caffe_gpu_mul(buffer_blob_.count(), top_data, buffer_data, top_data);

      // shift
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
          batch_sum_multiplier_.gpu_data(),
          this->blobs_[1]->gpu_data(), Dtype(0),
          spatial_mean_data);
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_, H_ * W_, 1,
          Dtype(1),
          spatial_mean_.gpu_data(), spatial_sum_multiplier_.gpu_data(), Dtype(0),
          buffer_blob_.mutable_gpu_data());
      caffe_gpu_add(buffer_blob_.count(), top_data, buffer_data, top_data);
      break;
    case BNParameter_BNMode_INFERENCE:
      // scale
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
          batch_sum_multiplier_.gpu_data(), this->blobs_[0]->gpu_data(),
          Dtype(0), spatial_variance_.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_,
          H_ * W_, 1, Dtype(1),
          spatial_variance_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
          Dtype(0), buffer_blob_.mutable_gpu_data());

      caffe_gpu_mul(buffer_blob_.count(), bottom_data, buffer_data, top_data);

      // shift
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
          batch_sum_multiplier_.gpu_data(),
          this->blobs_[1]->gpu_data(), Dtype(0),
          spatial_mean_data);
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_, H_ * W_, 1,
          Dtype(1),
          spatial_mean_.gpu_data(), spatial_sum_multiplier_.gpu_data(), Dtype(0),
          buffer_blob_.mutable_gpu_data());
      caffe_gpu_add(buffer_blob_.count(), top_data, buffer_data, top_data);
      break;
    default:
      LOG(FATAL) << "Unknown BN mode.";
    }
  }

  template <typename Dtype>
  void BNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* const_bottom_diff = bottom[0]->gpu_diff();
    Dtype* spatial_mean_data = spatial_mean_.mutable_gpu_data();
    Dtype* buffer_data = buffer_blob_.mutable_gpu_data();
    const Dtype* const_buffer_data = buffer_blob_.gpu_data();

    switch (this->layer_param_.bn_param().bn_mode()) {
    case BNParameter_BNMode_LEARN:
      // Propage to layer params
      // gradient w.r.t. scale
      caffe_gpu_mul(buffer_blob_.count(), x_norm_.gpu_data(),
          top_diff, buffer_blob_.mutable_gpu_data());
      // EX across spatial
      caffe_gpu_gemv<Dtype>(CblasNoTrans, N_ * C_, H_ * W_, Dtype(1),
          buffer_data, spatial_sum_multiplier_.gpu_data(), Dtype(0),
      spatial_variance_.mutable_gpu_data());
      // EX across batch
      caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_, Dtype(1),
          spatial_variance_.gpu_data(),
          batch_sum_multiplier_.gpu_data(), Dtype(0),
          this->blobs_[0]->mutable_gpu_diff());

      // gradient w.r.t. shift
      // EX across spatial
      caffe_gpu_gemv<Dtype>(CblasNoTrans, N_ * C_, H_ * W_, Dtype(1),
          top_diff, spatial_sum_multiplier_.gpu_data(),
          Dtype(0), spatial_mean_data);
      // EX across batch
      caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_, Dtype(1),
          spatial_mean_.gpu_data(),
          batch_sum_multiplier_.gpu_data(), Dtype(0),
          this->blobs_[1]->mutable_gpu_diff());

      // Propagate down
      // scale top diff
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
          batch_sum_multiplier_.gpu_data(), this->blobs_[0]->gpu_data(),
          Dtype(0), spatial_variance_.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_,
          H_ * W_, 1, Dtype(1),
          spatial_variance_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
          Dtype(0),
          buffer_blob_.mutable_gpu_data());
      caffe_gpu_mul(buffer_blob_.count(), top_diff, buffer_data,
          buffer_blob_.mutable_gpu_data());

      // use new top diff for computation
      caffe_gpu_mul(buffer_blob_.count(), x_norm_.gpu_data(),
          buffer_data, bottom_diff);
      // EX across spatial
      caffe_gpu_gemv<Dtype>(CblasNoTrans, N_ * C_, H_ * W_,
          Dtype(1), bottom_diff,
          spatial_sum_multiplier_.gpu_data(), Dtype(0), spatial_mean_data);
      // EX across batch
      caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_, Dtype(1),
          spatial_mean_.gpu_data(),
          batch_sum_multiplier_.gpu_data(), Dtype(0),
          batch_mean_.mutable_gpu_data());

      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
          batch_sum_multiplier_.gpu_data(),
          batch_mean_.gpu_data(), Dtype(0),
          spatial_mean_data);
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_,
          H_ * W_, 1, Dtype(1), spatial_mean_.gpu_data(),
          spatial_sum_multiplier_.gpu_data(), Dtype(0),
          bottom_diff);

      caffe_gpu_mul(buffer_blob_.count(), x_norm_.gpu_data(),
          bottom_diff, bottom_diff);

      // EX across spatial
      caffe_gpu_gemv<Dtype>(CblasNoTrans, N_ * C_, H_ * W_, Dtype(1),
          buffer_data, spatial_sum_multiplier_.gpu_data(),
          Dtype(0), spatial_mean_data);

      // EX across batch
      caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_, Dtype(1),
          spatial_mean_.gpu_data(),
          batch_sum_multiplier_.gpu_data(), Dtype(0),
          batch_mean_.mutable_gpu_data());

      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_,
          C_, 1, Dtype(1),
          batch_sum_multiplier_.gpu_data(),
          batch_mean_.gpu_data(), Dtype(0),
          spatial_mean_data);
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_,
          H_ * W_, 1, Dtype(1),
          spatial_mean_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
          Dtype(1),
          bottom_diff);

      caffe_gpu_axpby(buffer_blob_.count(), Dtype(1), buffer_data,
          Dtype(-1. / (N_ * H_ * W_)),
          bottom_diff);

      // put the squares of bottom into buffer_blob_
      caffe_gpu_powx(buffer_blob_.count(), bottom_data, Dtype(2),
          buffer_blob_.mutable_gpu_data());

      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
          batch_sum_multiplier_.gpu_data(), batch_variance_.gpu_data(), Dtype(0),
          spatial_variance_.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_,
          H_ * W_, 1, Dtype(1),
          spatial_variance_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
          Dtype(0),
          buffer_blob_.mutable_gpu_data());

      caffe_gpu_div(buffer_blob_.count(), const_bottom_diff,
          const_buffer_data, bottom_diff);
      break;
    case BNParameter_BNMode_INFERENCE:
      // Propage to layer params
      // gradient w.r.t. scale
      caffe_gpu_mul(buffer_blob_.count(), bottom_data,
          top_diff, buffer_blob_.mutable_gpu_data());
      // EX across spatial
      caffe_gpu_gemv<Dtype>(CblasNoTrans, N_ * C_, H_ * W_, Dtype(1),
          buffer_data, spatial_sum_multiplier_.gpu_data(), Dtype(0),
      spatial_variance_.mutable_gpu_data());
      // EX across batch
      caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_, Dtype(1),
          spatial_variance_.gpu_data(),
          batch_sum_multiplier_.gpu_data(), Dtype(0),
          this->blobs_[0]->mutable_gpu_diff());

      // gradient w.r.t. shift
      // EX across spatial
      caffe_gpu_gemv<Dtype>(CblasNoTrans, N_ * C_, H_ * W_, Dtype(1),
          top_diff, spatial_sum_multiplier_.gpu_data(),
          Dtype(0), spatial_mean_data);
      // EX across batch
      caffe_gpu_gemv<Dtype>(CblasTrans, N_, C_, Dtype(1),
          spatial_mean_.gpu_data(),
          batch_sum_multiplier_.gpu_data(), Dtype(0),
          this->blobs_[1]->mutable_gpu_diff());

      // Propagate down
      // scale top diff
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
          batch_sum_multiplier_.gpu_data(), this->blobs_[0]->gpu_data(),
          Dtype(0), spatial_variance_.mutable_gpu_data());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_ * C_,
          H_ * W_, 1, Dtype(1),
          spatial_variance_.gpu_data(), spatial_sum_multiplier_.gpu_data(),
          Dtype(0),
          buffer_blob_.mutable_gpu_data());
      caffe_gpu_mul(buffer_blob_.count(), top_diff, buffer_data,
          bottom_diff);
      break;
    default:
      LOG(FATAL) << "Unknown BN mode.";
    }
  }

  INSTANTIATE_LAYER_GPU_FUNCS(BNLayer);
}  // namespace caffe
