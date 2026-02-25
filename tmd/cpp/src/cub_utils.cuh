

namespace tmd {

// CUBSumOp is useful for CUB reduce functions such as
// cub::DeviceReduce::ReduceByKey
struct CUBSumOp {
  template <typename T>
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return a + b;
  }
};

} // namespace tmd
