#include <ATen/ATen.h>
#include <climits>
#include <cmath>
#include <cstdio>
#include <immintrin.h>
#include <iostream>
#include <sys/time.h>
#include <time.h>
#include <torch/extension.h>
#include <string>
#include <vector>

// Uncomment for ISPC
// #include "module_ispc.h"
// using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y,
                        const int &sizeX) {
  // Note that sizeX is the size of a Row, not the number of rows
  return tensor[x * (sizeX) + y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y,
                        const int &sizeX, float &val) {
  tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z,
                         int &b, const int &sizeX, const int &sizeY,
                         const int &sizeZ) {
  return tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * sizeZ + b];
}

inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z,
                         int &b, const int &sizeX, const int &sizeY,
                         const int &sizeZ, float &val) {
  tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * sizeZ + b] = val;
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
  tensor = tensor.flatten();
  tensor = tensor.contiguous();
  std::vector<float> vec(tensor.data_ptr<float>(),
                         tensor.data_ptr<float>() + tensor.numel());
  return vec;
}

/* Programming Your Attention Modules.
 *
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We
 * have also created O and QK^t Tensors that are formatted as vectors. After you
 * have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it
 * this way - if I asked my dnn a question and it output 5 different answers it
 * had a batch size of 5. These samples are independent of each other and thus
 * can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This
 * effectively allows each head to operate the same attention algorithm, but
 * each with each head using different hyperparameters. These allow each head to
 * have their own definition of what relevance is when looking at a token. These
 * heads can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the
 * number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per
 * attention head. Let's say I encoded a word using the follow (length, number
 * of vowels, has a capital letters). The emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor,
                               torch::Tensor VTensor, torch::Tensor QK_tTensor,
                               int B, int H, int N, int d) {

  // Q, K, V are passed in with Shape: (B, H, N, d)
  // QK^t Intermediate Tensor has Shape (N, N)

  // Make O Tensor with Shape (B, H, N, d)
  at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

  // Format O, Q, K, and V tensors into 4D vectors
  std::vector<float> O = formatTensor(OTensor);
  std::vector<float> Q = formatTensor(QTensor);
  std::vector<float> K = formatTensor(KTensor);
  std::vector<float> V = formatTensor(VTensor);
  std::fill(O.begin(), O.end(), 0);

  // Format QK_t Tensor into a 2D vector.
  std::vector<float> QK_t = formatTensor(QK_tTensor);

  /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D
     accessors

      //loop over Batch Size
       for (int b = 0; b < B; b++) {

           //loop over Heads
           for (int h = 0; h < H; h++) {

               //loop over Sequence Length
               for (int i = 0; i < N; i++) {

                   //loop over Embedding Dimensionality
                   for (int j = 0; j < d; j++) {
                      float val = fourDimRead(Q, b, h, i, j, H, N, d);
                      val = 0.0;
                      fourDimWrite(Q, b, h, i, j, H, N, d, val);
                   }
               }
           }
       }
  */

  /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D
     accessors

         for (int i = 0; i < N; i++) {
             for (int j = 0; j < N; j++) {
                 float val = twoDimRead(QK_t, i, j, N);
             val = 0.0;
                 twoDimWrite(QK_t, i, j, N, val);
           }
       }
  */

  // -------- YOUR CODE HERE  -------- //
  for (int batch_idx = 0; batch_idx < B; batch_idx++) {
    for (int head_idx = 0; head_idx < H; head_idx++) {
      for (int q_token_idx = 0; q_token_idx < N; q_token_idx++) {
        for (int k_token_idx = 0; k_token_idx < N; k_token_idx++) {
          float inner_product = 0;
          for (int i = 0; i < d; i++) {
            float q_val = fourDimRead(Q, batch_idx, head_idx, q_token_idx, i, H, N, d);
            float k_val = fourDimRead(K, batch_idx, head_idx, k_token_idx, i, H, N, d);
            inner_product += q_val * k_val;
          }
          twoDimWrite(QK_t, q_token_idx, k_token_idx, N, inner_product);
        } 
      }
      // softmax
      for (int i = 0; i < N; i++) {
        float exp_val_sum = 0;
        for (int j = 0; j < N; j++) {
          float exp_val = std::exp(twoDimRead(QK_t, i, j, N));
          exp_val_sum += exp_val;
          twoDimWrite(QK_t, i, j, N, exp_val);
        }
        for (int j = 0; j < N; j++) {
          float new_val = twoDimRead(QK_t, i, j, N) / exp_val_sum;
          twoDimWrite(QK_t, i, j, N, new_val);
        }
      }

      // QK_t * V
      for (int q_token_idx = 0; q_token_idx < N; q_token_idx++) {
        for (int kv_token_idx = 0; kv_token_idx < N; kv_token_idx++) {
          float attention_score = twoDimRead(QK_t, q_token_idx, kv_token_idx, N);
          // attention_score将对应token加权
          for (int dim_idx = 0; dim_idx < d; dim_idx++) {
            float new_val = attention_score * fourDimRead(V, batch_idx, head_idx, kv_token_idx, dim_idx, H, N, d);
            float O_old_val = fourDimRead(O, batch_idx, head_idx, q_token_idx, dim_idx, H, N, d);
            float O_new_val = new_val + O_old_val;
            fourDimWrite(O, batch_idx, head_idx, q_token_idx, dim_idx, H, N, d, O_new_val);
          }
        }
      }

    }
  }

  // DO NOT EDIT THIS RETURN STATEMENT //
  // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and
  // returns it //
  return torch::from_blob(O.data(), {B, H, N, d},
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}

// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor,
                                        torch::Tensor KTensor,
                                        torch::Tensor VTensor,
                                        torch::Tensor QK_tTensor, int B, int H,
                                        int N, int d) {

  // Q, K, V are passed in with Shape: (B, H, N, d)
  // QK^t Intermediate Tensor has Shape (N, N)

  // Make O Tensor with Shape (B, H, N, d)
  at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

  // Format O, Q, K, and V tensors into 4D vectors
  std::vector<float> O = formatTensor(OTensor);
  std::vector<float> Q = formatTensor(QTensor);
  std::vector<float> K = formatTensor(KTensor);
  std::vector<float> V = formatTensor(VTensor);

  // Format QK_t Tensor into a 2D vector.
  std::vector<float> QK_t = formatTensor(QK_tTensor);

  // -------- YOUR CODE HERE  -------- //
  for (int batch_idx = 0; batch_idx < B; batch_idx++) {
    for (int head_idx = 0; head_idx < H; head_idx++) {
      for (int q_token_idx = 0; q_token_idx < N; q_token_idx++) {
        for (int k_token_idx = 0; k_token_idx < N; k_token_idx++) {
          float inner_product = 0;
          for (int i = 0; i < d; i++) {
            float q_val = fourDimRead(Q, batch_idx, head_idx, q_token_idx, i, H, N, d);
            float k_val = fourDimRead(K, batch_idx, head_idx, k_token_idx, i, H, N, d);
            inner_product += q_val * k_val;
          }
          twoDimWrite(QK_t, q_token_idx, k_token_idx, N, inner_product);
        } 
      }
      // softmax
      for (int i = 0; i < N; i++) {
        float exp_val_sum = 0;
        for (int j = 0; j < N; j++) {
          float exp_val = std::exp(twoDimRead(QK_t, i, j, N));
          exp_val_sum += exp_val;
          twoDimWrite(QK_t, i, j, N, exp_val);
        }
        for (int j = 0; j < N; j++) {
          float new_val = twoDimRead(QK_t, i, j, N) / exp_val_sum;
          twoDimWrite(QK_t, i, j, N, new_val);
        }
      }


      // QK_t * V
      for (int q_token_idx = 0; q_token_idx < N; q_token_idx++) {
        for (int kv_token_idx = 0; kv_token_idx < N; kv_token_idx++) {
          float attention_score = twoDimRead(QK_t, q_token_idx, kv_token_idx, N);
          // attention_score将对应token加权
          for (int dim_idx = 0; dim_idx < d; dim_idx++) {
            float new_val = attention_score * fourDimRead(V, batch_idx, head_idx, kv_token_idx, dim_idx, H, N, d);
            float O_old_val = fourDimRead(O, batch_idx, head_idx, q_token_idx, dim_idx, H, N, d);
            float O_new_val = new_val + O_old_val;
            fourDimWrite(O, batch_idx, head_idx, q_token_idx, dim_idx, H, N, d, O_new_val);
          }
        }
      }
    }
  }

  // DO NOT EDIT THIS RETURN STATEMENT //
  // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and
  // returns it //
  return torch::from_blob(O.data(), {B, H, N, d},
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}

// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor,
                               torch::Tensor VTensor, torch::Tensor temp, int B,
                               int H, int N, int d) {

  // Q, K, V are passed in with Shape: (B, H, N, d)

  // Make O Tensor with Shape (B, H, N, d)
  // and O Row Tensor with Shape (N)
  at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
  at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

  // Format Y, Q, K, and V tensors into 4D vectors
  std::vector<float> O = formatTensor(OTensor);
  std::vector<float> Q = formatTensor(QTensor);
  std::vector<float> K = formatTensor(KTensor);
  std::vector<float> V = formatTensor(VTensor);

  // Format ORow Tensor into a 1D vector
  //  You can simply access this as ORow[i]
  std::vector<float> ORow = formatTensor(ORowTensor);

  // -------- YOUR CODE HERE  -------- //
  #pragma omp parallel for collapse(3)
  for (int batch_idx = 0; batch_idx < B; batch_idx++) {
    for (int head_idx = 0; head_idx < H; head_idx++) {
      for (int q_token_idx = 0; q_token_idx < N; q_token_idx++) {
        // YRow is moved inside so each OpenMP thread gets a local copy.
        at::Tensor ORowTensor = temp.index({torch::indexing::Slice(
            omp_get_thread_num(), torch::indexing::None)});
        std::vector<float> ORow = formatTensor(ORowTensor);
        for (int k_token_idx = 0; k_token_idx < N; k_token_idx++) {
          float inner_product = 0;
          for (int i = 0; i < d; i++) {
            float q_val = fourDimRead(Q, batch_idx, head_idx, q_token_idx, i, H, N, d);
            float k_val = fourDimRead(K, batch_idx, head_idx, k_token_idx, i, H, N, d);
            inner_product += q_val * k_val;
          }
          ORow[k_token_idx] = inner_product;
        } 
        float max = 0;
        for (int i = 0; i < N; i++) {
          max = ORow[i] > max ? ORow[i] : max;
        }
        // softmax
        float exp_val_sum = 0;
        for (int i = 0; i < N; i++) {
          float exp_val = std::exp(ORow[i] - max);
          exp_val_sum += exp_val;
          ORow[i] = exp_val;
        }
        for (int i = 0; i < N; i++) {
          // 每算出来一个softmax值之后可以直接乘以V的一行
          float attention_score = ORow[i] / exp_val_sum;
          for (int dim_idx = 0; dim_idx < d; dim_idx++) {
            float new_val = attention_score * fourDimRead(V, batch_idx, head_idx, i, dim_idx, H, N, d);
            float O_old_val = fourDimRead(O, batch_idx, head_idx, q_token_idx, dim_idx, H, N, d);
            float O_new_val = new_val + O_old_val;
            fourDimWrite(O, batch_idx, head_idx, q_token_idx, dim_idx, H, N, d, O_new_val);
          }
        }
      }
    }
  }
  // DO NOT EDIT THIS RETURN STATEMENT //
  // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and
  // returns it //
  return torch::from_blob(O.data(), {B, H, N, d},
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}
// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION V1		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttentionV1(torch::Tensor QTensor, torch::Tensor KTensor,
                               torch::Tensor VTensor, torch::Tensor QiTensor,
                               torch::Tensor KjTensor, torch::Tensor VjTensor,
                               torch::Tensor SijTensor, torch::Tensor PijTensor,
                               torch::Tensor PVTensor, torch::Tensor OiTensor,
                               torch::Tensor LTensor, torch::Tensor LiTensor,
                               torch::Tensor LijTensor,
                               torch::Tensor LnewTensor, int Bc, int Br, int B,
                               int H, int N, int d) {

  // Q, K, V are passed in with Shape: (B, H, N, d)
  // Sij, Pij are passed in with Shape: (Br, Bc)
  // Kj, Vj are passed in with Shape: (Bc, d)
  // Qi, Oi, and PV  are passed in with Shape: (Br, d)
  // L in passed in with Shape: (N)
  // Li, Lij, and Lnew are passed in with shape (Br)

  // Make O Tensor with Shape (B, H, N, d)
  at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
  // Format All Tensors into Vectors
  std::vector<float> O = formatTensor(OTensor);
  std::vector<float> Q = formatTensor(QTensor);
  std::vector<float> K = formatTensor(KTensor);
  std::vector<float> V = formatTensor(VTensor);
  std::vector<float> Sij = formatTensor(SijTensor);
  std::vector<float> Pij = formatTensor(PijTensor);
  std::vector<float> Kj = formatTensor(KjTensor);
  std::vector<float> Vj = formatTensor(VjTensor);
  std::vector<float> Qi = formatTensor(QiTensor);
  std::vector<float> Oi = formatTensor(OiTensor);
  // std::vector<float> l = formatTensor(LTensor);
  // std::vector<float> PV = formatTensor(PVTensor);
  std::vector<float> li = formatTensor(LiTensor);
  std::vector<float> Lij = formatTensor(LijTensor);
  std::vector<float> lnew = formatTensor(LnewTensor);

  // 向上取整计算Tc和Tr
  int Tc = (N + Bc - 1)/ Bc;
  int Tr = (N + Br - 1) / Br;
  int kv_block_size = Bc;
  int q_block_size = Br;
  std::vector<float> Mij(q_block_size);
  std::vector<float> Mi_new(q_block_size);
  std::vector<float> M(N, INT_MIN);
  std::vector<float> L(N, 0.0);
  std::vector<float> PV(d, 0.0);
  // Lij Lnew Mij Mi_new Sij Pij都是局部的，而L、M、Q、K、V、O是全局的；二者的访问
  // -------- YOUR CODE HERE  -------- //
  #pragma omp parallel for collapse(2) firstprivate(Mij, Mi_new, M, L, PV, lnew, Lij, Sij, Pij)
  for (int batch_idx = 0; batch_idx < B; batch_idx++) {
    for (int head_idx = 0; head_idx < H; head_idx++) {
      // std::fill(M.begin(), M.end(), INT_MIN);
      // std::fill(L.begin(), L.end(), 0.0);
      // std::fill(PV.begin(), PV.end(), 0.0);
      for (int kv_block_idx = 0; kv_block_idx < Tc; kv_block_idx++) {
        for (int q_block_idx = 0; q_block_idx < Tr; q_block_idx++) {
          int q_block_start_idx = q_block_idx * q_block_size;
          int kv_block_start_idx = kv_block_idx * kv_block_size;
          q_block_size = std::min(q_block_size, N - q_block_start_idx);
          kv_block_size = std::min(kv_block_size, N - kv_block_start_idx);
          for (int q_token_idx = q_block_start_idx; q_token_idx < q_block_start_idx + q_block_size; q_token_idx++) {
            int max = INT_MIN; 
            int q_local_idx = q_token_idx - q_block_start_idx;
            for (int k_token_idx = kv_block_start_idx; k_token_idx < kv_block_start_idx + kv_block_size; k_token_idx++) {
              int k_local_idx = k_token_idx - kv_block_start_idx;
              float inner_product = 0;
              for (int i = 0; i < d; i++) {
                float q_val = fourDimRead(Q, batch_idx, head_idx, q_token_idx, i, H, N, d);
                float k_val = fourDimRead(K, batch_idx, head_idx, k_token_idx, i, H, N, d);
                inner_product += q_val * k_val;
              }
              twoDimWrite(Sij, q_local_idx, k_local_idx, kv_block_size, inner_product);
              max = max > inner_product ? max : inner_product; 
            }
            Mij[q_local_idx] = max;
            float Pij_sum = 0;
            for (int k_local_idx = 0; k_local_idx < kv_block_size; k_local_idx++) {
              float Sij_val = twoDimRead(Sij, q_local_idx, k_local_idx, kv_block_size);
              float new_Sij_val = std::exp(Sij_val - Mij[q_local_idx]);
              twoDimWrite(Pij, q_local_idx, k_local_idx, kv_block_size, new_Sij_val);
              Pij_sum += new_Sij_val;
            }
            Lij[q_local_idx] = Pij_sum;
            Mi_new[q_local_idx] = std::max(Mij[q_local_idx], M[q_token_idx]);
            float exp_Mi_minus_MiNew = std::exp(M[q_token_idx] - Mi_new[q_local_idx]);
            float exp_Mij_minus_MiNew = std::exp(Mij[q_local_idx] - Mi_new[q_local_idx]);
            lnew[q_local_idx] = exp_Mi_minus_MiNew * L[q_token_idx] + exp_Mij_minus_MiNew * Lij[q_local_idx];
            // Pij * Vj
            for (int k_token_idx = kv_block_start_idx; k_token_idx < kv_block_start_idx + kv_block_size; k_token_idx++) {
              int k_local_idx = k_token_idx - kv_block_start_idx;
              float Pij_val = twoDimRead(Pij, q_local_idx, k_local_idx, kv_block_size);
              for (int dim_idx = 0; dim_idx <d; dim_idx++) {
                float new_val = Pij_val * fourDimRead(V, batch_idx, head_idx, k_token_idx, dim_idx, H, N, d);
                PV[dim_idx] += new_val;
              } 
            }
            for (int dim_idx = 0; dim_idx < d; dim_idx++) {
              float old_O = fourDimRead(O, batch_idx, head_idx, q_token_idx, dim_idx, H, N, d);
              float new_O = (L[q_token_idx] * exp_Mi_minus_MiNew * old_O + exp_Mij_minus_MiNew * PV[dim_idx]) / lnew[q_local_idx];
              fourDimWrite(O, batch_idx, head_idx, q_token_idx, dim_idx, H, N, d, new_O);
              PV[dim_idx] = 0.0;
            }
            L[q_token_idx] = lnew[q_local_idx];
            M[q_token_idx] = Mi_new[q_local_idx];
          }
        }
      }
    }
  }

  // DO NOT EDIT THIS RETURN STATEMENT //
  // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and
  // returns it //
  return torch::from_blob(O.data(), {B, H, N, d},
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}

// ---------------------------------------------------------- //
//                PART 5: FLASH ATTENTION V2		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor,
                               torch::Tensor VTensor, torch::Tensor QiTensor,
                               torch::Tensor KjTensor, torch::Tensor VjTensor,
                               torch::Tensor SijTensor, torch::Tensor PijTensor,
                               torch::Tensor PVTensor, torch::Tensor OiTensor,
                               torch::Tensor LTensor, torch::Tensor LiTensor,
                               torch::Tensor LijTensor,
                               torch::Tensor LnewTensor, int Bc, int Br, int B,
                               int H, int N, int d) {

  // Q, K, V are passed in with Shape: (B, H, N, d)
  // Sij, Pij are passed in with Shape: (Br, Bc)
  // Kj, Vj are passed in with Shape: (Bc, d)
  // Qi, Oi, and PV  are passed in with Shape: (Br, d)
  // L in passed in with Shape: (N)
  // Li, Lij, and Lnew are passed in with shape (Br)

  // Make O Tensor with Shape (B, H, N, d)
  at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
  // Format All Tensors into Vectors
  std::vector<float> O = formatTensor(OTensor);
  std::vector<float> Q = formatTensor(QTensor);
  std::vector<float> K = formatTensor(KTensor);
  std::vector<float> V = formatTensor(VTensor);
  std::vector<float> Sij = formatTensor(SijTensor);
  std::vector<float> Pij = formatTensor(PijTensor);
  std::vector<float> Kj = formatTensor(KjTensor);
  std::vector<float> Vj = formatTensor(VjTensor);
  std::vector<float> Qi = formatTensor(QiTensor);
  std::vector<float> Oi = formatTensor(OiTensor);
  // std::vector<float> l = formatTensor(LTensor);
  // std::vector<float> PV = formatTensor(PVTensor);
  std::vector<float> li = formatTensor(LiTensor);
  std::vector<float> Lij = formatTensor(LijTensor);
  std::vector<float> lnew = formatTensor(LnewTensor);

  // 向上取整计算Tc和Tr
  int Tc = (N + Bc - 1)/ Bc;
  int Tr = (N + Br - 1) / Br;
  int kv_block_size = Bc;
  int q_block_size = Br;
  std::vector<float> Mij(q_block_size);
  std::vector<float> Mi_new(q_block_size);
  std::vector<float> M(N, INT_MIN);
  std::vector<float> L(N, 0.0);
  std::vector<float> PV(d, 0.0);
  // -------- YOUR CODE HERE  -------- //
  omp_set_nested(1);
  #pragma omp parallel for collapse(3) firstprivate(Mij, Mi_new, M, L, PV, lnew, Lij, Sij, Pij)
  for (int batch_idx = 0; batch_idx < B; batch_idx++) {
    for (int head_idx = 0; head_idx < H; head_idx++) {
      for (int q_block_idx = 0; q_block_idx < Tr; q_block_idx++) {

        for (int kv_block_idx = 0; kv_block_idx < Tc; kv_block_idx++) {
          int q_block_start_idx = q_block_idx * q_block_size;
          int kv_block_start_idx = kv_block_idx * kv_block_size;
          q_block_size = std::min(q_block_size, N - q_block_start_idx);
          kv_block_size = std::min(kv_block_size, N - kv_block_start_idx);

          #pragma omp parallel for firstprivate(PV)
          for (int q_token_idx = q_block_start_idx; q_token_idx < q_block_start_idx + q_block_size; q_token_idx++) {
            int max = INT_MIN; 
            int q_local_idx = q_token_idx - q_block_start_idx;
            for (int k_token_idx = kv_block_start_idx; k_token_idx < kv_block_start_idx + kv_block_size; k_token_idx++) {
              int k_local_idx = k_token_idx - kv_block_start_idx;
              float inner_product = 0;
              for (int i = 0; i < d; i++) {
                float q_val = fourDimRead(Q, batch_idx, head_idx, q_token_idx, i, H, N, d);
                float k_val = fourDimRead(K, batch_idx, head_idx, k_token_idx, i, H, N, d);
                inner_product += q_val * k_val;
              }
              twoDimWrite(Sij, q_local_idx, k_local_idx, kv_block_size, inner_product);
              max = max > inner_product ? max : inner_product; 
            }
            // 存储q与当前的key block的局部attention score的最大值
            Mij[q_local_idx] = max;
            float Pij_sum = 0;
            // 计算q与当前key block的局部attention score的softmax的分母
            for (int k_local_idx = 0; k_local_idx < kv_block_size; k_local_idx++) {
              float Sij_val = twoDimRead(Sij, q_local_idx, k_local_idx, kv_block_size);
              float new_Sij_val = std::exp(Sij_val - Mij[q_local_idx]);
              twoDimWrite(Pij, q_local_idx, k_local_idx, kv_block_size, new_Sij_val);
              Pij_sum += new_Sij_val;
            }
            Lij[q_local_idx] = Pij_sum;
            // 到目前为止q的attention score的最大值
            Mi_new[q_local_idx] = std::max(Mij[q_local_idx], M[q_token_idx]);
            float exp_Mi_minus_MiNew = std::exp(M[q_token_idx] - Mi_new[q_local_idx]);
            float exp_Mij_minus_MiNew = std::exp(Mij[q_local_idx] - Mi_new[q_local_idx]);
            lnew[q_local_idx] = exp_Mi_minus_MiNew * L[q_token_idx] + exp_Mij_minus_MiNew * Lij[q_local_idx];
            // Pij * Vj
            for (int k_token_idx = kv_block_start_idx; k_token_idx < kv_block_start_idx + kv_block_size; k_token_idx++) {
              int k_local_idx = k_token_idx - kv_block_start_idx;
              float Pij_val = twoDimRead(Pij, q_local_idx, k_local_idx, kv_block_size);
              for (int dim_idx = 0; dim_idx <d; dim_idx++) {
                float new_val = Pij_val * fourDimRead(V, batch_idx, head_idx, k_token_idx, dim_idx, H, N, d);
                PV[dim_idx] += new_val;
              } 
            }
            for (int dim_idx = 0; dim_idx < d; dim_idx++) {
              float old_O = fourDimRead(O, batch_idx, head_idx, q_token_idx, dim_idx, H, N, d);
              float new_O = (L[q_token_idx] * exp_Mi_minus_MiNew * old_O + exp_Mij_minus_MiNew * PV[dim_idx]) / lnew[q_local_idx];
              fourDimWrite(O, batch_idx, head_idx, q_token_idx, dim_idx, H, N, d, new_O);
              PV[dim_idx] = 0.0;
            }
            L[q_token_idx] = lnew[q_local_idx];
            M[q_token_idx] = Mi_new[q_local_idx];
          }
        }
      }
    }
  }

  // DO NOT EDIT THIS RETURN STATEMENT //
  // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and
  // returns it //
  return torch::from_blob(O.data(), {B, H, N, d},
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}
/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked,
        " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
