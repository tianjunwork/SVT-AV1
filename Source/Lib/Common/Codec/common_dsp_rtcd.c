/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

/*
* Copyright (c) 2016, Alliance for Open Media. All rights reserved
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at www.aomedia.org/license/software. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at www.aomedia.org/license/patent.
*/

#include "common_dsp_rtcd.h"
#include "EbPictureOperators.h"
#include "EbPackUnPack_C.h"
#include "EbAvcStyleMcp.h"

/*
 * DSP deprecated flags
 */
#define HAS_MMX CPU_FLAGS_MMX
#define HAS_SSE CPU_FLAGS_SSE
#define HAS_SSE2 CPU_FLAGS_SSE2
#define HAS_SSE3 CPU_FLAGS_SSE3
#define HAS_SSSE3 CPU_FLAGS_SSSE3
#define HAS_SSE4_1 CPU_FLAGS_SSE4_1
#define HAS_SSE4_2 CPU_FLAGS_SSE4_2
#define HAS_AVX CPU_FLAGS_AVX
#define HAS_AVX2 CPU_FLAGS_AVX2
#define HAS_AVX512F CPU_FLAGS_AVX512F
#define HAS_AVX512CD CPU_FLAGS_AVX512CD
#define HAS_AVX512DQ CPU_FLAGS_AVX512DQ
#define HAS_AVX512ER CPU_FLAGS_AVX512ER
#define HAS_AVX512PF CPU_FLAGS_AVX512PF
#define HAS_AVX512BW CPU_FLAGS_AVX512BW
#define HAS_AVX512VL CPU_FLAGS_AVX512VL


/**********************************************
 * global function pointer variable definition 
 **********************************************/
void(*aom_blend_a64_vmask)(uint8_t *dst, uint32_t dst_stride, const uint8_t *src0, uint32_t src0_stride, const uint8_t *src1, uint32_t src1_stride, const uint8_t *mask, int w, int h);
void(*aom_highbd_blend_a64_vmask)(uint8_t *dst, uint32_t dst_stride, const uint8_t *src0, uint32_t src0_stride, const uint8_t *src1, uint32_t src1_stride, const uint8_t *mask, int w, int h, int bd);
void(*aom_highbd_blend_a64_hmask)(uint8_t *dst, uint32_t dst_stride, const uint8_t *src0, uint32_t src0_stride, const uint8_t *src1, uint32_t src1_stride, const uint8_t *mask, int w, int h, int bd);
void(*aom_blend_a64_hmask)(uint8_t *dst, uint32_t dst_stride, const uint8_t *src0, uint32_t src0_stride, const uint8_t *src1, uint32_t src1_stride, const uint8_t *mask, int w, int h);
void(*aom_blend_a64_mask)(uint8_t *dst, uint32_t dst_stride, const uint8_t *src0, uint32_t src0_stride, const uint8_t *src1, uint32_t src1_stride, const uint8_t *mask, uint32_t mask_stride, int w, int h, int subx, int suby);
void(*aom_highbd_blend_a64_mask)(uint8_t *dst, uint32_t dst_stride, const uint8_t *src0, uint32_t src0_stride, const uint8_t *src1, uint32_t src1_stride, const uint8_t *mask, uint32_t mask_stride, int w, int h, int subx, int suby, int bd);
void(*eb_aom_highbd_blend_a64_vmask)(uint16_t *dst, uint32_t dst_stride, const uint16_t *src0, uint32_t src0_stride, const uint16_t *src1, uint32_t src1_stride, const uint8_t *mask, int w, int h, int bd);
void(*eb_aom_highbd_blend_a64_hmask)(uint16_t *dst, uint32_t dst_stride, const uint16_t *src0, uint32_t src0_stride, const uint16_t *src1, uint32_t src1_stride, const uint8_t *mask, int w, int h, int bd);
void(*eb_cfl_predict_lbd)(const int16_t *pred_buf_q3, uint8_t *pred, int32_t pred_stride, uint8_t *dst, int32_t dst_stride, int32_t alpha_q3, int32_t bit_depth, int32_t width, int32_t height);
void(*eb_cfl_predict_hbd)(const int16_t *pred_buf_q3, uint16_t *pred, int32_t pred_stride, uint16_t *dst, int32_t dst_stride, int32_t alpha_q3, int32_t bit_depth, int32_t width, int32_t height);
void(*cfl_luma_subsampling_420_lbd)(const uint8_t *input, int32_t input_stride, int16_t *output_q3, int32_t width, int32_t height);
void(*cfl_luma_subsampling_420_hbd)(const uint16_t *input, int32_t input_stride, int16_t *output_q3, int32_t width, int32_t height);
void(*eb_av1_filter_intra_predictor) (uint8_t *dst, ptrdiff_t stride, TxSize tx_size, const uint8_t *above, const uint8_t *left, int32_t mode);
void(*eb_av1_filter_intra_edge)(uint8_t *p, int32_t sz, int32_t strength);
void(*eb_av1_filter_intra_edge_high)(uint16_t *p, int32_t sz, int32_t strength);
void(*eb_av1_upsample_intra_edge)(uint8_t *p, int32_t sz);
//void(*eb_av1_upsample_intra_edge_high)(uint16_t *p, int32_t sz, int32_t bd);
void(*eb_av1_highbd_dr_prediction_z2)(uint16_t *dst, ptrdiff_t stride, int32_t bw, int32_t bh, const uint16_t *above, const uint16_t *left, int32_t upsample_above, int32_t upsample_left, int32_t dx, int32_t dy, int32_t bd);
void(*av1_build_compound_diffwtd_mask_d16)(uint8_t *mask, DIFFWTD_MASK_TYPE mask_type, const CONV_BUF_TYPE *src0, int src0_stride, const CONV_BUF_TYPE *src1, int src1_stride, int h, int w, ConvolveParams *conv_params, int bd);
void(*eb_av1_inv_txfm2d_add_4x4)(const int32_t *input, uint16_t *output_r, int32_t stride_r, uint16_t *output_w, int32_t stride_w, TxType tx_type, int32_t bd);
void(*eb_av1_inv_txfm2d_add_8x8)(const int32_t *input, uint16_t *output_r, int32_t stride_r, uint16_t *output_w, int32_t stride_w, TxType tx_type, int32_t bd);
void(*eb_av1_inv_txfm2d_add_16x16)(const int32_t *input, uint16_t *output_r, int32_t stride_r, uint16_t *output_w, int32_t stride_w, TxType tx_type, int32_t bd);
void(*eb_av1_inv_txfm2d_add_32x32)(const int32_t *input, uint16_t *output_r, int32_t stride_r, uint16_t *output_w, int32_t stride_w, TxType tx_type, int32_t bd);
void(*eb_av1_inv_txfm2d_add_64x64)(const int32_t *input, uint16_t *output_r, int32_t stride_r, uint16_t *output_w, int32_t stride_w, TxType tx_type, int32_t bd);
void(*eb_av1_inv_txfm2d_add_8x16)(const int32_t *input, uint16_t *output_r, int32_t stride_r, uint16_t *output_w, int32_t stride_w, TxType tx_type, TxSize tx_size, int32_t eob, int32_t bd);
void(*eb_av1_inv_txfm2d_add_16x8)(const int32_t *input, uint16_t *output_r, int32_t stride_r, uint16_t *output_w, int32_t stride_we, TxType tx_type, TxSize tx_size, int32_t eob, int32_t bd);
void(*eb_av1_inv_txfm2d_add_16x32)(const int32_t *input, uint16_t *output_r, int32_t stride_r, uint16_t *output_w, int32_t stride_w, TxType tx_type, TxSize tx_size, int32_t eob, int32_t bd);
void(*eb_av1_inv_txfm2d_add_32x16)(const int32_t *input, uint16_t *output_r, int32_t stride_r, uint16_t *output_w, int32_t stride_w, TxType tx_type, TxSize tx_size, int32_t eob, int32_t bd);
void(*eb_av1_inv_txfm2d_add_32x8)(const int32_t *input, uint16_t *output_r, int32_t stride_r, uint16_t *output_w, int32_t stride_w, TxType tx_type, TxSize tx_size, int32_t eob, int32_t bd);
void(*eb_av1_inv_txfm2d_add_8x32)(const int32_t *input, uint16_t *output_r, int32_t stride_r, uint16_t *output_w, int32_t stride_w, TxType tx_type, TxSize tx_size, int32_t eob, int32_t bd);
void(*eb_av1_inv_txfm2d_add_32x64)(const int32_t *input, uint16_t *output_r, int32_t stride_r, uint16_t *output_w, int32_t stride_w, TxType tx_type, TxSize tx_size, int32_t eob, int32_t bd);
void(*eb_av1_inv_txfm2d_add_64x32)(const int32_t *input, uint16_t *output_r, int32_t stride_r, uint16_t *output_w, int32_t stride_w, TxType tx_type, TxSize tx_size, int32_t eob, int32_t bd);
void(*eb_av1_inv_txfm2d_add_16x64)(const int32_t *input, uint16_t *output_r, int32_t stride_r, uint16_t *output_w, int32_t stride_w, TxType tx_type, TxSize tx_size, int32_t eob, int32_t bd);
void(*eb_av1_inv_txfm2d_add_64x16)(const int32_t *input, uint16_t *output_r, int32_t stride_r, uint16_t *output_w, int32_t stride_w, TxType tx_type, TxSize tx_size, int32_t eob, int32_t bd);
void(*eb_av1_inv_txfm2d_add_4x8)(const int32_t *input, uint16_t *output_r, int32_t stride_r, uint16_t *output_w, int32_t stride_w, TxType tx_type, TxSize tx_size, int32_t bd);
void(*eb_av1_inv_txfm2d_add_8x4)(const int32_t *input, uint16_t *output_r, int32_t stride_r, uint16_t *output_w, int32_t stride_w, TxType tx_type, TxSize tx_size, int32_t bd);
void(*eb_av1_inv_txfm2d_add_4x16)(const int32_t *input, uint16_t *output_r, int32_t stride_r, uint16_t *output_w, int32_t stride_w, TxType tx_type, TxSize tx_size, int32_t bd);
void(*eb_av1_inv_txfm2d_add_16x4)(const int32_t *input, uint16_t *output_r, int32_t stride_r, uint16_t *output_w, int32_t stride_w, TxType tx_type, TxSize tx_size, int32_t bd);
void(*eb_av1_inv_txfm_add)(const TranLow *dqcoeff, uint8_t *dst_r, int32_t stride_r, uint8_t *dst_w, int32_t stride_w, const TxfmParam *txfm_param);
void(*compressed_packmsb)(uint8_t *in8_bit_buffer, uint32_t in8_stride, uint8_t *inn_bit_buffer, uint16_t *out16_bit_buffer, uint32_t inn_stride, uint32_t out_stride, uint32_t width, uint32_t height);
void(*c_pack)(const uint8_t *inn_bit_buffer, uint32_t inn_stride, uint8_t *in_compn_bit_buffer, uint32_t out_stride, uint8_t *local_cache, uint32_t width, uint32_t height);
void(*unpack_avg)(uint16_t *ref16_l0, uint32_t ref_l0_stride, uint16_t *ref16_l1, uint32_t ref_l1_stride, uint8_t *dst_ptr, uint32_t dst_stride, uint32_t width, uint32_t height);
void(*unpack_avg_safe_sub)(uint16_t *ref16_l0, uint32_t ref_l0_stride, uint16_t *ref16_l1, uint32_t ref_l1_stride, uint8_t *dst_ptr, uint32_t dst_stride, EbBool sub_pred, uint32_t width, uint32_t height);
void(*un_pack8_bit_data)(uint16_t *in16_bit_buffer, uint32_t in_stride, uint8_t *out8_bit_buffer, uint32_t out8_stride, uint32_t width, uint32_t height);
void(*convert_8bit_to_16bit)(uint8_t *src, uint32_t src_stride, uint16_t *dst, uint32_t dst_stride, uint32_t width, uint32_t height);
void(*convert_16bit_to_8bit)(uint16_t *src, uint32_t src_stride, uint8_t *dst, uint32_t dst_stride, uint32_t width, uint32_t height);
void(*pack2d_16_bit_src_mul4)(uint8_t *in8_bit_buffer, uint32_t in8_stride, uint8_t *inn_bit_buffer, uint16_t *out16_bit_buffer, uint32_t inn_stride, uint32_t out_stride, uint32_t width, uint32_t height);
void(*un_pack2d_16_bit_src_mul4)(uint16_t *in16_bit_buffer, uint32_t in_stride, uint8_t *out8_bit_buffer, uint8_t *outn_bit_buffer, uint32_t out8_stride, uint32_t outn_stride, uint32_t width, uint32_t height);
void(*residual_kernel8bit)(uint8_t *input, uint32_t input_stride, uint8_t *pred, uint32_t pred_stride, int16_t *residual, uint32_t residual_stride, uint32_t area_width, uint32_t area_height);
uint64_t(*compute8x8_satd_u8)(uint8_t *diff, uint64_t *dc_value, uint32_t src_stride);
int32_t(*sum_residual8bit)(int16_t *in_ptr, uint32_t size, uint32_t stride_in);
void(*full_distortion_kernel_cbf_zero32_bits)(int32_t *coeff, uint32_t coeff_stride, int32_t *recon_coeff, uint32_t recon_coeff_stride, uint64_t distortion_result[DIST_CALC_TOTAL], uint32_t area_width, uint32_t area_height);
void(*full_distortion_kernel32_bits)(int32_t *coeff, uint32_t coeff_stride, int32_t *recon_coeff, uint32_t recon_coeff_stride, uint64_t distortion_result[DIST_CALC_TOTAL], uint32_t area_width, uint32_t area_height);
void(*picture_average_kernel)(EbByte src0, uint32_t src0_stride, EbByte src1, uint32_t src1_stride, EbByte dst, uint32_t dst_stride, uint32_t area_width, uint32_t area_height);
void(*picture_average_kernel1_line)(EbByte src0, EbByte src1, EbByte dst, uint32_t area_width);
uint64_t(*spatial_full_distortion_kernel)(uint8_t *input, uint32_t input_offset, uint32_t input_stride, uint8_t *recon, int32_t recon_offset, uint32_t recon_stride, uint32_t area_width, uint32_t area_height);
uint64_t(*full_distortion_kernel16_bits)(uint8_t* input, uint32_t input_offset, uint32_t input_stride, uint8_t* recon, int32_t recon_offset, uint32_t recon_stride, uint32_t area_width, uint32_t area_height);
void(*residual_kernel16bit)(uint16_t *input, uint32_t input_stride, uint16_t *pred, uint32_t pred_stride, int16_t *residual, uint32_t residual_stride, uint32_t area_width, uint32_t area_height);
void(*avc_style_luma_interpolation_filter)(EbByte ref_pic, uint32_t src_stride, EbByte dst, uint32_t dst_stride, uint32_t pu_width, uint32_t pu_height, EbByte temp_buf, EbBool skip, uint32_t frac_pos, uint8_t choice);
void(*eb_av1_wiener_convolve_add_src)(const uint8_t *const src, const ptrdiff_t src_stride, uint8_t *const dst, const ptrdiff_t dst_stride, const int16_t *const filter_x, const int16_t *const filter_y, const int32_t w, const int32_t h, const ConvolveParams *const conv_params);
void(*eb_av1_highbd_wiener_convolve_add_src)(const uint8_t *const src, const ptrdiff_t src_stride, uint8_t *const dst, const ptrdiff_t dst_stride, const int16_t *const filter_x, const int16_t *const filter_y, const int32_t w, const int32_t h, const ConvolveParams *const conv_params, const int32_t bd);
void(*eb_apply_selfguided_restoration)(const uint8_t *dat, int32_t width, int32_t height, int32_t stride, int32_t eps, const int32_t *xqd, uint8_t *dst, int32_t dst_stride, int32_t *tmpbuf, int32_t bit_depth, int32_t highbd);
void(*eb_av1_selfguided_restoration)(const uint8_t *dgd8, int32_t width, int32_t height, int32_t dgd_stride, int32_t *flt0, int32_t *flt1, int32_t flt_stride, int32_t sgr_params_idx, int32_t bit_depth, int32_t highbd);
void(*eb_av1_convolve_2d_copy_sr)(const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride, int32_t w, int32_t h, InterpFilterParams *filter_params_x, InterpFilterParams *filter_params_y, const int32_t subpel_x_q4, const int32_t subpel_y_q4, ConvolveParams *conv_params);
void(*eb_av1_convolve_2d_sr)(const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride, int32_t w, int32_t h, InterpFilterParams *filter_params_x, InterpFilterParams *filter_params_y, const int32_t subpel_x_q4, const int32_t subpel_y_q4, ConvolveParams *conv_params);
void(*eb_av1_jnt_convolve_2d_copy)(const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride, int32_t w, int32_t h, InterpFilterParams *filter_params_x, InterpFilterParams *filter_params_y, const int32_t subpel_x_q4, const int32_t subpel_y_q4, ConvolveParams *conv_params);
void(*eb_av1_convolve_x_sr)(const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride, int32_t w, int32_t h, InterpFilterParams *filter_params_x, InterpFilterParams *filter_params_y, const int32_t subpel_x_q4, const int32_t subpel_y_q4, ConvolveParams *conv_params);
void(*eb_av1_convolve_y_sr)(const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride, int32_t w, int32_t h, InterpFilterParams *filter_params_x, InterpFilterParams *filter_params_y, const int32_t subpel_x_q4, const int32_t subpel_y_q4, ConvolveParams *conv_params);
void(*eb_av1_convolve_2d_scale)(const uint8_t *src, int src_stride, uint8_t *dst, int dst_stride, int w, int h, const InterpFilterParams *filter_params_x, const InterpFilterParams *filter_params_y, const int subpel_x_qn, const int x_step_qn, const int subpel_y_q4, const int y_step_qn, ConvolveParams *conv_params);
void(*eb_av1_jnt_convolve_x)(const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride, int32_t w, int32_t h, InterpFilterParams *filter_params_x, InterpFilterParams *filter_params_y, const int32_t subpel_x_q4, const int32_t subpel_y_q4, ConvolveParams *conv_params);
void(*eb_av1_jnt_convolve_y)(const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride, int32_t w, int32_t h, InterpFilterParams *filter_params_x, InterpFilterParams *filter_params_y, const int32_t subpel_x_q4, const int32_t subpel_y_q4, ConvolveParams *conv_params);
void(*eb_av1_jnt_convolve_2d)(const uint8_t *src, int32_t src_stride, uint8_t *dst, int32_t dst_stride, int32_t w, int32_t h, InterpFilterParams *filter_params_x, InterpFilterParams *filter_params_y, const int32_t subpel_x_q4, const int32_t subpel_y_q4, ConvolveParams *conv_params);
void(*eb_av1_highbd_convolve_2d_copy_sr)(const uint16_t *src, int32_t src_stride, uint16_t *dst, int32_t dst_stride, int32_t w, int32_t h, const InterpFilterParams *filter_params_x, const InterpFilterParams *filter_params_y, const int32_t subpel_x_q4, const int32_t subpel_y_q4, ConvolveParams *conv_params, int32_t bd);
void(*eb_av1_highbd_jnt_convolve_2d_copy)(const uint16_t *src, int32_t src_stride, uint16_t *dst, int32_t dst_stride, int32_t w, int32_t h, const InterpFilterParams *filter_params_x, const InterpFilterParams *filter_params_y, const int32_t subpel_x_q4, const int32_t subpel_y_q4, ConvolveParams *conv_params, int32_t bd);
void(*eb_av1_highbd_convolve_y_sr)(const uint16_t *src, int32_t src_stride, uint16_t *dst, int32_t dst_stride, int32_t w, int32_t h, const InterpFilterParams *filter_params_x, const InterpFilterParams *filter_params_y, const int32_t subpel_x_q4, const int32_t subpel_y_q4, ConvolveParams *conv_params, int32_t bd);
void(*eb_av1_highbd_convolve_2d_sr)(const uint16_t *src, int32_t src_stride, uint16_t *dst, int32_t dst_stride, int32_t w, int32_t h, const InterpFilterParams *filter_params_x, const InterpFilterParams *filter_params_y, const int32_t subpel_x_q4, const int32_t subpel_y_q4, ConvolveParams *conv_params, int32_t bd);
void(*eb_av1_highbd_convolve_2d_scale)(const uint16_t *src, int src_stride, uint16_t *dst, int dst_stride, int w, int h, const InterpFilterParams *filter_params_x, const InterpFilterParams *filter_params_y, const int subpel_x_q4, const int x_step_qn, const int subpel_y_q4, const int y_step_qn, ConvolveParams *conv_params, int bd);
void(*eb_av1_highbd_jnt_convolve_2d)(const uint16_t *src, int32_t src_stride, uint16_t *dst, int32_t dst_stride, int32_t w, int32_t h, const InterpFilterParams *filter_params_x, const InterpFilterParams *filter_params_y, const int32_t subpel_x_q4, const int32_t subpel_y_q4, ConvolveParams *conv_params, int32_t bd);
void(*eb_av1_highbd_jnt_convolve_x)(const uint16_t *src, int32_t src_stride, uint16_t *dst, int32_t dst_stride, int32_t w, int32_t h, const InterpFilterParams *filter_params_x, const InterpFilterParams *filter_params_y, const int32_t subpel_x_q4, const int32_t subpel_y_q4, ConvolveParams *conv_params, int32_t bd);
void(*eb_av1_highbd_jnt_convolve_y)(const uint16_t *src, int32_t src_stride, uint16_t *dst, int32_t dst_stride, int32_t w, int32_t h, const InterpFilterParams *filter_params_x, const InterpFilterParams *filter_params_y, const int32_t subpel_x_q4, const int32_t subpel_y_q4, ConvolveParams *conv_params, int32_t bd);
void(*eb_av1_highbd_convolve_x_sr)(const uint16_t *src, int32_t src_stride, uint16_t *dst, int32_t dst_stride, int32_t w, int32_t h, const InterpFilterParams *filter_params_x, const InterpFilterParams *filter_params_y, const int32_t subpel_x_q4, const int32_t subpel_y_q4, ConvolveParams *conv_params, int32_t bd);
void(*aom_convolve8_horiz)(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
void(*aom_convolve8_vert)(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst, ptrdiff_t dst_stride, const int16_t *filter_x, int x_step_q4, const int16_t *filter_y, int y_step_q4, int w, int h);
void(*av1_build_compound_diffwtd_mask)(uint8_t *mask, DIFFWTD_MASK_TYPE mask_type, const uint8_t *src0, int src0_stride, const uint8_t *src1, int src1_stride, int h, int w);
void(*av1_build_compound_diffwtd_mask_highbd)(uint8_t *mask, DIFFWTD_MASK_TYPE mask_type, const uint8_t *src0, int src0_stride, const uint8_t *src1, int src1_stride, int h, int w, int bd);
uint64_t(*av1_wedge_sse_from_residuals)(const int16_t *r1, const int16_t *d, const uint8_t *m, int N);
void(*aom_subtract_block)(int rows, int cols, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr, ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride);
void(*aom_highbd_subtract_block) (int rows, int cols, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr, ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride, int bd);
void(*aom_lowbd_blend_a64_d16_mask)(uint8_t *dst, uint32_t dst_stride, const CONV_BUF_TYPE *src0, uint32_t src0_stride, const CONV_BUF_TYPE *src1, uint32_t src1_stride, const uint8_t *mask, uint32_t mask_stride, int w, int h, int subw, int subh, ConvolveParams *conv_params);
void(*aom_highbd_blend_a64_d16_mask)(uint8_t *dst, uint32_t dst_stride, const CONV_BUF_TYPE *src0, uint32_t src0_stride, const CONV_BUF_TYPE *src1, uint32_t src1_stride, const uint8_t *mask, uint32_t mask_stride, int w, int h, int subx, int suby, ConvolveParams *conv_params, const int bd);
void(*eb_aom_highbd_dc_128_predictor_16x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_128_predictor_16x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_128_predictor_16x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_128_predictor_16x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_128_predictor_16x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_128_predictor_32x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_128_predictor_32x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_128_predictor_32x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_128_predictor_32x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_128_predictor_4x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_128_predictor_4x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_128_predictor_4x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_128_predictor_64x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_128_predictor_64x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_128_predictor_64x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_128_predictor_8x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_128_predictor_8x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_128_predictor_8x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_128_predictor_8x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_left_predictor_16x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_left_predictor_16x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_left_predictor_16x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_left_predictor_16x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_left_predictor_16x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_left_predictor_2x2)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_left_predictor_32x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_left_predictor_32x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_left_predictor_32x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_left_predictor_32x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_left_predictor_4x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_left_predictor_4x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_left_predictor_4x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_left_predictor_64x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_left_predictor_64x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_left_predictor_64x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_left_predictor_8x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_left_predictor_8x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_left_predictor_8x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_left_predictor_8x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_predictor_16x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_predictor_16x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_predictor_16x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_predictor_16x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_predictor_16x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_predictor_2x2)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_predictor_32x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_predictor_32x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_predictor_32x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_predictor_32x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_predictor_4x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_predictor_4x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_predictor_4x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_predictor_64x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_predictor_64x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_predictor_64x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_predictor_8x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_predictor_8x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_predictor_8x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_predictor_8x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_top_predictor_16x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_top_predictor_16x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_top_predictor_16x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_top_predictor_16x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_top_predictor_16x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_top_predictor_32x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_top_predictor_32x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_top_predictor_32x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_top_predictor_32x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_top_predictor_4x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_top_predictor_4x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_top_predictor_4x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_top_predictor_64x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_top_predictor_64x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_top_predictor_64x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_top_predictor_8x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_top_predictor_8x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_top_predictor_8x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_dc_top_predictor_8x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_h_predictor_16x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_h_predictor_16x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_h_predictor_16x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_h_predictor_16x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_h_predictor_16x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
// void(*eb_aom_highbd_h_predictor_2x2)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_h_predictor_32x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_h_predictor_32x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_h_predictor_32x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_h_predictor_32x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_h_predictor_4x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_h_predictor_4x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_h_predictor_4x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_h_predictor_64x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_h_predictor_64x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_h_predictor_64x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_h_predictor_8x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_h_predictor_8x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_h_predictor_8x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_h_predictor_8x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_h_predictor_16x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_h_predictor_16x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_h_predictor_16x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_h_predictor_16x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_h_predictor_16x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_h_predictor_32x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_h_predictor_32x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_h_predictor_32x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_h_predictor_32x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_h_predictor_4x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_h_predictor_4x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_h_predictor_4x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_h_predictor_64x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_h_predictor_64x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_h_predictor_64x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_h_predictor_8x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_h_predictor_8x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_h_predictor_8x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_h_predictor_8x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_predictor_16x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_predictor_16x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_predictor_16x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_predictor_16x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_predictor_16x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_predictor_2x2)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_predictor_32x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_predictor_32x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_predictor_32x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_predictor_32x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_predictor_4x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_predictor_4x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_predictor_4x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_predictor_64x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_predictor_64x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_predictor_64x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_predictor_8x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_predictor_8x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_predictor_8x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_predictor_8x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_v_predictor_16x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_v_predictor_16x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_v_predictor_16x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_v_predictor_16x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_v_predictor_16x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_v_predictor_2x2)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_v_predictor_32x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_v_predictor_32x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_v_predictor_32x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_v_predictor_32x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_v_predictor_4x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_v_predictor_4x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_v_predictor_4x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_v_predictor_64x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_v_predictor_64x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_v_predictor_64x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_v_predictor_8x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_v_predictor_8x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_v_predictor_8x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_smooth_v_predictor_8x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_v_predictor_16x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_v_predictor_16x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_v_predictor_16x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_v_predictor_16x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_v_predictor_16x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_v_predictor_32x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_v_predictor_32x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_v_predictor_32x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_v_predictor_32x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_v_predictor_4x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_v_predictor_4x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_v_predictor_4x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_v_predictor_64x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_v_predictor_64x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_v_predictor_64x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_v_predictor_8x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_v_predictor_8x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_v_predictor_8x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_aom_highbd_v_predictor_8x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int32_t bd);
void(*eb_av1_dr_prediction_z1)(uint8_t *dst, ptrdiff_t stride, int32_t bw, int32_t bh, const uint8_t *above, const uint8_t *left, int32_t upsample_above, int32_t dx, int32_t dy);
void(*eb_av1_dr_prediction_z2)(uint8_t *dst, ptrdiff_t stride, int32_t bw, int32_t bh, const uint8_t *above, const uint8_t *left, int32_t upsample_above, int32_t upsample_left, int32_t dx, int32_t dy);
void(*eb_av1_dr_prediction_z3)(uint8_t *dst, ptrdiff_t stride, int32_t bw, int32_t bh, const uint8_t *above, const uint8_t *left, int32_t upsample_left, int32_t dx, int32_t dy);
void(*eb_av1_highbd_dr_prediction_z1)(uint16_t *dst, ptrdiff_t stride, int32_t bw, int32_t bh, const uint16_t *above, const uint16_t *left, int32_t upsample_above, int32_t dx, int32_t dy, int32_t bd);
void(*eb_av1_highbd_dr_prediction_z3)(uint16_t *dst, ptrdiff_t stride, int32_t bw, int32_t bh, const uint16_t *above, const uint16_t *left, int32_t upsample_left, int32_t dx, int32_t dy, int32_t bd);
void(*eb_aom_dc_predictor_4x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_predictor_8x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_predictor_16x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_predictor_32x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_predictor_64x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_predictor_16x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_predictor_16x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_predictor_16x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_predictor_16x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_predictor_32x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_predictor_32x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_predictor_32x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_predictor_4x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_predictor_4x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_predictor_64x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_predictor_64x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_predictor_8x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_predictor_8x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_predictor_8x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_left_predictor_4x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_left_predictor_8x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_left_predictor_16x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_left_predictor_32x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_left_predictor_64x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_left_predictor_16x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_left_predictor_16x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_left_predictor_16x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_left_predictor_16x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_left_predictor_32x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_left_predictor_32x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_left_predictor_32x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_left_predictor_4x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_left_predictor_4x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_left_predictor_64x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_left_predictor_64x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_left_predictor_8x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_left_predictor_8x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_left_predictor_8x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_top_predictor_4x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_top_predictor_8x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_top_predictor_16x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_top_predictor_32x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_top_predictor_64x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_top_predictor_16x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_top_predictor_16x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_top_predictor_16x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_top_predictor_16x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_top_predictor_32x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_top_predictor_32x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_top_predictor_32x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_top_predictor_4x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_top_predictor_4x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_top_predictor_64x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_top_predictor_64x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_top_predictor_8x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_top_predictor_8x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_top_predictor_8x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_128_predictor_4x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_128_predictor_8x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_128_predictor_16x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_128_predictor_32x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_128_predictor_64x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_128_predictor_16x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_128_predictor_16x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_128_predictor_16x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_128_predictor_16x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_128_predictor_32x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_128_predictor_32x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_128_predictor_32x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_128_predictor_4x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_128_predictor_4x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_128_predictor_64x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_128_predictor_64x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_128_predictor_8x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_128_predictor_8x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_dc_128_predictor_8x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_h_predictor_64x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_h_predictor_32x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_h_predictor_16x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_h_predictor_8x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_h_predictor_4x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_h_predictor_16x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_h_predictor_16x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_h_predictor_16x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_h_predictor_16x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_h_predictor_32x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_h_predictor_32x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_h_predictor_32x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_h_predictor_4x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_h_predictor_4x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_h_predictor_64x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_h_predictor_64x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_h_predictor_8x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_h_predictor_8x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_h_predictor_8x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_v_predictor_64x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_v_predictor_32x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_v_predictor_16x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_v_predictor_8x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_v_predictor_4x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_v_predictor_16x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_v_predictor_16x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_v_predictor_16x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_v_predictor_16x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_v_predictor_32x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_v_predictor_32x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_v_predictor_32x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_v_predictor_4x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_v_predictor_4x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_v_predictor_64x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_v_predictor_64x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_v_predictor_8x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_v_predictor_8x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_v_predictor_8x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_predictor_64x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_predictor_32x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_predictor_16x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_predictor_8x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_predictor_4x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_predictor_16x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_predictor_16x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_predictor_16x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_predictor_16x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_predictor_32x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_predictor_32x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_predictor_32x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_predictor_4x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_predictor_4x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_predictor_64x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_predictor_64x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_predictor_8x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_predictor_8x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_smooth_predictor_8x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_v_predictor_4x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_v_predictor_8x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_v_predictor_16x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_v_predictor_32x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_v_predictor_64x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_v_predictor_16x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_v_predictor_16x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_v_predictor_16x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_v_predictor_16x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_v_predictor_32x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_v_predictor_32x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_v_predictor_32x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_v_predictor_4x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_v_predictor_4x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_v_predictor_64x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_v_predictor_64x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_v_predictor_8x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_v_predictor_8x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_v_predictor_8x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_h_predictor_4x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_h_predictor_8x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_h_predictor_16x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_h_predictor_32x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_h_predictor_64x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_h_predictor_16x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_h_predictor_16x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_h_predictor_16x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_h_predictor_16x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_h_predictor_32x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_h_predictor_32x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_h_predictor_32x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_h_predictor_4x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_h_predictor_4x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_h_predictor_64x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_h_predictor_64x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_h_predictor_8x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_h_predictor_8x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_h_predictor_8x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_paeth_predictor_16x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_paeth_predictor_16x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_paeth_predictor_16x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_paeth_predictor_16x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_paeth_predictor_16x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_paeth_predictor_32x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_paeth_predictor_32x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_paeth_predictor_32x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_paeth_predictor_32x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_paeth_predictor_4x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_paeth_predictor_4x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_paeth_predictor_4x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_paeth_predictor_64x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_paeth_predictor_64x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_paeth_predictor_64x64)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_paeth_predictor_8x16)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_paeth_predictor_8x32)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_paeth_predictor_8x4)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_paeth_predictor_8x8)(uint8_t *dst, ptrdiff_t y_stride, const uint8_t *above, const uint8_t *left);
void(*eb_aom_highbd_paeth_predictor_16x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd);
void(*eb_aom_highbd_paeth_predictor_16x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd);
void(*eb_aom_highbd_paeth_predictor_16x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd);
void(*eb_aom_highbd_paeth_predictor_16x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd);
void(*eb_aom_highbd_paeth_predictor_16x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd);
void(*eb_aom_highbd_paeth_predictor_2x2)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd);
void(*eb_aom_highbd_paeth_predictor_32x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd);
void(*eb_aom_highbd_paeth_predictor_32x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd);
void(*eb_aom_highbd_paeth_predictor_32x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd);
void(*eb_aom_highbd_paeth_predictor_32x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd);
void(*eb_aom_highbd_paeth_predictor_4x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd);
void(*eb_aom_highbd_paeth_predictor_4x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd);
void(*eb_aom_highbd_paeth_predictor_4x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd);
void(*eb_aom_highbd_paeth_predictor_64x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd);
void(*eb_aom_highbd_paeth_predictor_64x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd);
void(*eb_aom_highbd_paeth_predictor_64x64)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd);
void(*eb_aom_highbd_paeth_predictor_8x16)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd);
void(*eb_aom_highbd_paeth_predictor_8x32)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd);
void(*eb_aom_highbd_paeth_predictor_8x4)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd);
void(*eb_aom_highbd_paeth_predictor_8x8)(uint16_t *dst, ptrdiff_t y_stride, const uint16_t *above, const uint16_t *left, int bd);
uint64_t(*aom_sum_squares_i16)(const int16_t *src, uint32_t N);
int32_t(*eb_cdef_find_dir)(const uint16_t *img, int32_t stride, int32_t *var, int32_t coeff_shift);
void(*eb_cdef_filter_block)(uint8_t *dst8, uint16_t *dst16, int32_t dstride, const uint16_t *in, int32_t pri_strength, int32_t sec_strength, int32_t dir, int32_t pri_damping, int32_t sec_damping, int32_t bsize, int32_t coeff_shift);
void(*eb_cdef_filter_block_8x8_16)(const uint16_t *const in, const int32_t pri_strength, const int32_t sec_strength, const int32_t dir, int32_t pri_damping, int32_t sec_damping, const int32_t coeff_shift, uint16_t *const dst, const int32_t dstride);
void(*eb_copy_rect8_8bit_to_16bit)(uint16_t *dst, int32_t dstride, const uint8_t *src, int32_t sstride, int32_t v, int32_t h);
void(*eb_av1_highbd_warp_affine)(const int32_t *mat, const uint16_t *ref, int width, int height, int stride, uint16_t *pred, int p_col, int p_row, int p_width, int p_height, int p_stride, int subsampling_x, int subsampling_y, int bd, ConvolveParams *conv_params, int16_t alpha, int16_t beta, int16_t gamma, int16_t delta);
void(*eb_av1_warp_affine)(const int32_t *mat, const uint8_t *ref, int width, int height, int stride, uint8_t *pred, int p_col, int p_row, int p_width, int p_height, int p_stride, int subsampling_x, int subsampling_y, ConvolveParams *conv_params, int16_t alpha, int16_t beta, int16_t gamma, int16_t delta);
void(*aom_highbd_lpf_horizontal_14)(uint16_t *s, int32_t pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh, int32_t bd);
void(*aom_highbd_lpf_horizontal_4)(uint16_t *s, int32_t pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh, int32_t bd);
void(*aom_highbd_lpf_horizontal_6)(uint16_t *s, int32_t pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh, int32_t bd);
void(*aom_highbd_lpf_horizontal_8)(uint16_t *s, int32_t pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh, int32_t bd);
void(*aom_highbd_lpf_vertical_14)(uint16_t *s, int32_t pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh, int32_t bd);
void(*aom_highbd_lpf_vertical_4)(uint16_t *s, int32_t pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh, int32_t bd);
void(*aom_highbd_lpf_vertical_6)(uint16_t *s, int32_t pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh, int32_t bd);
void(*aom_highbd_lpf_vertical_8)(uint16_t *s, int32_t pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh, int32_t bd);
void(*aom_lpf_horizontal_14)(uint8_t *s, int32_t pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
void(*aom_lpf_horizontal_4)(uint8_t *s, int32_t pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
void(*aom_lpf_horizontal_6)(uint8_t *s, int32_t pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
void(*aom_lpf_horizontal_8)(uint8_t *s, int32_t pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
void(*aom_lpf_vertical_14)(uint8_t *s, int32_t pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
void(*aom_lpf_vertical_4)(uint8_t *s, int32_t pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
void(*aom_lpf_vertical_6)(uint8_t *s, int32_t pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
void(*aom_lpf_vertical_8)(uint8_t *s, int32_t pitch, const uint8_t *blimit, const uint8_t *limit, const uint8_t *thresh);
uint32_t(*eb_log2f)(uint32_t x);
void(*eb_memcpy)(void  *dst_ptr, void  *src_ptr, size_t size);

/**************************************
 * Instruction Set Support
 **************************************/
#ifdef ARCH_X86
// Helper Functions
static INLINE void run_cpuid(uint32_t eax, uint32_t ecx, uint32_t* abcd) {
#ifdef _WIN32
    __cpuidex(abcd, eax, ecx);
#else
    uint32_t ebx = 0, edx = 0;
#if defined(__i386__) && defined(__PIC__)
    /* in case of PIC under 32-bit EBX cannot be clobbered */
    __asm__("movl %%ebx, %%edi \n\t cpuid \n\t xchgl %%ebx, %%edi"
            : "=D"(ebx),
#else
    __asm__("cpuid"
    : "+b"(ebx),
#endif
    "+a"(eax),
    "+c"(ecx),
    "=d"(edx));
    abcd[0] = eax;
    abcd[1] = ebx;
    abcd[2] = ecx;
    abcd[3] = edx;
#endif
}

static INLINE int32_t check_xcr0_ymm() {
    uint32_t xcr0;
#ifdef _WIN32
    xcr0 = (uint32_t)_xgetbv(0); /* min VS2010 SP1 compiler is required */
#else
    __asm__("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");
#endif
    return ((xcr0 & 6) == 6); /* checking if xmm and ymm state are enabled in XCR0 */
}

static int32_t check_4thgen_intel_core_features() {
    uint32_t abcd[4];
    uint32_t fma_movbe_osxsave_mask = ((1 << 12) | (1 << 22) | (1 << 27));
    uint32_t avx2_bmi12_mask        = (1 << 5) | (1 << 3) | (1 << 8);

    /* CPUID.(EAX=01H, ECX=0H):ECX.FMA[bit 12]==1   &&
    CPUID.(EAX=01H, ECX=0H):ECX.MOVBE[bit 22]==1 &&
    CPUID.(EAX=01H, ECX=0H):ECX.OSXSAVE[bit 27]==1 */
    run_cpuid(1, 0, abcd);
    if ((abcd[2] & fma_movbe_osxsave_mask) != fma_movbe_osxsave_mask) return 0;

    if (!check_xcr0_ymm()) return 0;

    /*  CPUID.(EAX=07H, ECX=0H):EBX.AVX2[bit 5]==1  &&
    CPUID.(EAX=07H, ECX=0H):EBX.BMI1[bit 3]==1  &&
    CPUID.(EAX=07H, ECX=0H):EBX.BMI2[bit 8]==1  */
    run_cpuid(7, 0, abcd);
    if ((abcd[1] & avx2_bmi12_mask) != avx2_bmi12_mask) return 0;
    /* CPUID.(EAX=80000001H):ECX.LZCNT[bit 5]==1 */
    run_cpuid(0x80000001, 0, abcd);
    if ((abcd[2] & (1 << 5)) == 0) return 0;
    return 1;
}

static INLINE int check_xcr0_zmm() {
    uint32_t xcr0;
    uint32_t zmm_ymm_xmm = (7 << 5) | (1 << 2) | (1 << 1);
#ifdef _WIN32
    xcr0 = (uint32_t)_xgetbv(0); /* min VS2010 SP1 compiler is required */
#else
    __asm__("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");
#endif
    return ((xcr0 & zmm_ymm_xmm) ==
            zmm_ymm_xmm); /* check if xmm, ymm and zmm state are enabled in XCR0 */
}

static int32_t can_use_intel_avx512() {
    uint32_t abcd[4];

    /*  CPUID.(EAX=07H, ECX=0):EBX[bit 16]==1 AVX512F
    CPUID.(EAX=07H, ECX=0):EBX[bit 17] AVX512DQ
    CPUID.(EAX=07H, ECX=0):EBX[bit 28] AVX512CD
    CPUID.(EAX=07H, ECX=0):EBX[bit 30] AVX512BW
    CPUID.(EAX=07H, ECX=0):EBX[bit 31] AVX512VL */

    uint32_t avx512_ebx_mask = (1u << 16) // AVX-512F
                             | (1u << 17) // AVX-512DQ
                             | (1u << 28) // AVX-512CD
                             | (1u << 30) // AVX-512BW
                             | (1u << 31); // AVX-512VL

    if (!check_4thgen_intel_core_features()) return 0;

    // ensure OS supports ZMM registers (and YMM, and XMM)
    if (!check_xcr0_zmm()) return 0;

    run_cpuid(7, 0, abcd);
    if ((abcd[1] & avx512_ebx_mask) != avx512_ebx_mask) return 0;

    return 1;
}

CPU_FLAGS get_cpu_flags() {
    CPU_FLAGS flags = 0;

    /* To detail tests CPU features, requires more accurate implementation.
        Documentation help:
        https://docs.microsoft.com/en-us/cpp/intrinsics/cpuid-cpuidex?redirectedfrom=MSDN&view=vs-2019
    */

    if (check_4thgen_intel_core_features()) {
        flags |= CPU_FLAGS_MMX | CPU_FLAGS_SSE | CPU_FLAGS_SSE2 | CPU_FLAGS_SSE3 | CPU_FLAGS_SSSE3 |
                 CPU_FLAGS_SSE4_1 | CPU_FLAGS_SSE4_2 | CPU_FLAGS_AVX | CPU_FLAGS_AVX2;
    }

    if (can_use_intel_avx512()) {
        flags |= CPU_FLAGS_AVX512F | CPU_FLAGS_AVX512DQ | CPU_FLAGS_AVX512CD | CPU_FLAGS_AVX512BW |
                 CPU_FLAGS_AVX512VL;
    }

    return flags;
}

CPU_FLAGS get_cpu_flags_to_use() {
    CPU_FLAGS flags = get_cpu_flags();
#ifdef NON_AVX512_SUPPORT
    /* Remove AVX512 flags. */
    flags &= (CPU_FLAGS_AVX512F - 1);
#endif
    return flags;
}
#endif
#ifndef NON_AVX512_SUPPORT
#define SET_FUNCTIONS(ptr, c, mmx, sse, sse2, sse3, ssse3, sse4_1, sse4_2, avx, avx2, avx512) \
    do {                                                                                      \
        ptr = c;                                                                              \
        if (((uintptr_t)NULL != (uintptr_t)mmx) && (flags & HAS_MMX)) ptr = mmx;              \
        if (((uintptr_t)NULL != (uintptr_t)sse) && (flags & HAS_SSE)) ptr = sse;              \
        if (((uintptr_t)NULL != (uintptr_t)sse2) && (flags & HAS_SSE2)) ptr = sse2;           \
        if (((uintptr_t)NULL != (uintptr_t)sse3) && (flags & HAS_SSE3)) ptr = sse3;           \
        if (((uintptr_t)NULL != (uintptr_t)ssse3) && (flags & HAS_SSSE3)) ptr = ssse3;        \
        if (((uintptr_t)NULL != (uintptr_t)sse4_1) && (flags & HAS_SSE4_1)) ptr = sse4_1;     \
        if (((uintptr_t)NULL != (uintptr_t)sse4_2) && (flags & HAS_SSE4_2)) ptr = sse4_2;     \
        if (((uintptr_t)NULL != (uintptr_t)avx) && (flags & HAS_AVX)) ptr = avx;              \
        if (((uintptr_t)NULL != (uintptr_t)avx2) && (flags & HAS_AVX2)) ptr = avx2;           \
        if (((uintptr_t)NULL != (uintptr_t)avx512) && (flags & HAS_AVX512F)) ptr = avx512;    \
    } while (0)
#else
#define SET_FUNCTIONS(ptr, c, mmx, sse, sse2, sse3, ssse3, sse4_1, sse4_2, avx, avx2, avx512) \
    do {                                                                                      \
        ptr = c;                                                                              \
        if (((uintptr_t)NULL != (uintptr_t)mmx) && (flags & HAS_MMX)) ptr = mmx;              \
        if (((uintptr_t)NULL != (uintptr_t)sse) && (flags & HAS_SSE)) ptr = sse;              \
        if (((uintptr_t)NULL != (uintptr_t)sse2) && (flags & HAS_SSE2)) ptr = sse2;           \
        if (((uintptr_t)NULL != (uintptr_t)sse3) && (flags & HAS_SSE3)) ptr = sse3;           \
        if (((uintptr_t)NULL != (uintptr_t)ssse3) && (flags & HAS_SSSE3)) ptr = ssse3;        \
        if (((uintptr_t)NULL != (uintptr_t)sse4_1) && (flags & HAS_SSE4_1)) ptr = sse4_1;     \
        if (((uintptr_t)NULL != (uintptr_t)sse4_2) && (flags & HAS_SSE4_2)) ptr = sse4_2;     \
        if (((uintptr_t)NULL != (uintptr_t)avx) && (flags & HAS_AVX)) ptr = avx;              \
        if (((uintptr_t)NULL != (uintptr_t)avx2) && (flags & HAS_AVX2)) ptr = avx2;           \
    } while (0)
#endif

#define SET_SSE2(ptr, c, sse2) SET_FUNCTIONS(ptr, c, 0, 0, sse2, 0, 0, 0, 0, 0, 0, 0)
#define SET_SSE2_AVX2(ptr, c, sse2, avx2) SET_FUNCTIONS(ptr, c, 0, 0, sse2, 0, 0, 0, 0, 0, avx2, 0)
#define SET_SSSE3(ptr, c, ssse3) SET_FUNCTIONS(ptr, c, 0, 0, 0, 0, ssse3, 0, 0, 0, 0, 0)
#define SET_SSE41(ptr, c, sse4_1) SET_FUNCTIONS(ptr, c, 0, 0, 0, 0, 0, sse4_1, 0, 0, 0, 0)
#define SET_SSE41(ptr, c, sse4_1) SET_FUNCTIONS(ptr, c, 0, 0, 0, 0, 0, sse4_1, 0, 0, 0, 0)
#define SET_SSE41_AVX2(ptr, c, sse4_1, avx2) \
    SET_FUNCTIONS(ptr, c, 0, 0, 0, 0, 0, sse4_1, 0, 0, avx2, 0)
#define SET_SSE41_AVX2_AVX512(ptr, c, sse4_1, avx2, avx512) \
    SET_FUNCTIONS(ptr, c, 0, 0, 0, 0, 0, sse4_1, 0, 0, avx2, avx512)
#define SET_AVX2(ptr, c, avx2) SET_FUNCTIONS(ptr, c, 0, 0, 0, 0, 0, 0, 0, 0, avx2, 0)
#define SET_AVX2_AVX512(ptr, c, avx2, avx512) \
    SET_FUNCTIONS(ptr, c, 0, 0, 0, 0, 0, 0, 0, 0, avx2, avx512)


void setup_common_rtcd_internal(CPU_FLAGS flags) {
    /** Should be done during library initialization,
        but for safe limiting cpu flags again. */

    //to use C: flags=0
    (void)flags;
    aom_blend_a64_mask = aom_blend_a64_mask_c;
    aom_blend_a64_hmask = aom_blend_a64_hmask_c;
    aom_blend_a64_vmask = aom_blend_a64_vmask_c;

    aom_highbd_blend_a64_mask = aom_highbd_blend_a64_mask_c;
    aom_highbd_blend_a64_hmask = aom_highbd_blend_a64_hmask_c;
    aom_highbd_blend_a64_vmask = aom_highbd_blend_a64_vmask_c;

    eb_aom_highbd_blend_a64_vmask = eb_aom_highbd_blend_a64_vmask_c;
    eb_aom_highbd_blend_a64_hmask = eb_aom_highbd_blend_a64_hmask_c;

    eb_cfl_predict_lbd = eb_cfl_predict_lbd_c;
    eb_cfl_predict_hbd = eb_cfl_predict_hbd_c;

    eb_av1_filter_intra_predictor = eb_av1_filter_intra_predictor_c;

    eb_av1_filter_intra_edge_high = eb_av1_filter_intra_edge_high_c;

    eb_av1_filter_intra_edge = eb_av1_filter_intra_edge_high_c_old;

    eb_av1_upsample_intra_edge = eb_av1_upsample_intra_edge_c;

    av1_build_compound_diffwtd_mask_d16 = av1_build_compound_diffwtd_mask_d16_c;

    eb_av1_highbd_wiener_convolve_add_src = eb_av1_highbd_wiener_convolve_add_src_c;

    eb_apply_selfguided_restoration = eb_apply_selfguided_restoration_c;

    eb_av1_selfguided_restoration = eb_av1_selfguided_restoration_c;

    eb_av1_inv_txfm2d_add_16x16 = eb_av1_inv_txfm2d_add_16x16_c;
    eb_av1_inv_txfm2d_add_32x32 = eb_av1_inv_txfm2d_add_32x32_c;
    eb_av1_inv_txfm2d_add_4x4 = eb_av1_inv_txfm2d_add_4x4_c;
    eb_av1_inv_txfm2d_add_64x64 = eb_av1_inv_txfm2d_add_64x64_c;
    eb_av1_inv_txfm2d_add_8x8 = eb_av1_inv_txfm2d_add_8x8_c;

    eb_av1_inv_txfm2d_add_8x16 = eb_av1_inv_txfm2d_add_8x16_c;
    eb_av1_inv_txfm2d_add_16x8 = eb_av1_inv_txfm2d_add_16x8_c;
    eb_av1_inv_txfm2d_add_16x32 = eb_av1_inv_txfm2d_add_16x32_c;
    eb_av1_inv_txfm2d_add_32x16 = eb_av1_inv_txfm2d_add_32x16_c;
    eb_av1_inv_txfm2d_add_32x8 = eb_av1_inv_txfm2d_add_32x8_c;
    eb_av1_inv_txfm2d_add_8x32 = eb_av1_inv_txfm2d_add_8x32_c;
    eb_av1_inv_txfm2d_add_32x64 = eb_av1_inv_txfm2d_add_32x64_c;
    eb_av1_inv_txfm2d_add_64x32 = eb_av1_inv_txfm2d_add_64x32_c;
    eb_av1_inv_txfm2d_add_16x64 = eb_av1_inv_txfm2d_add_16x64_c;
    eb_av1_inv_txfm2d_add_64x16 = eb_av1_inv_txfm2d_add_64x16_c;
    eb_av1_inv_txfm2d_add_4x8 = eb_av1_inv_txfm2d_add_4x8_c;
    eb_av1_inv_txfm2d_add_8x4 = eb_av1_inv_txfm2d_add_8x4_c;
    eb_av1_inv_txfm2d_add_4x16 = eb_av1_inv_txfm2d_add_4x16_c;
    eb_av1_inv_txfm2d_add_16x4 = eb_av1_inv_txfm2d_add_16x4_c;

    eb_av1_inv_txfm_add = eb_av1_inv_txfm_add_c;

    compressed_packmsb = compressed_packmsb_c;
    c_pack = c_pack_c;
    unpack_avg = unpack_avg_c;
    unpack_avg_safe_sub = unpack_avg_safe_sub_c;
    un_pack8_bit_data = un_pack8_bit_data_c;
    cfl_luma_subsampling_420_lbd = cfl_luma_subsampling_420_lbd_c;
    cfl_luma_subsampling_420_hbd = cfl_luma_subsampling_420_hbd_c;
    convert_8bit_to_16bit = convert_8bit_to_16bit_c;
    convert_16bit_to_8bit = convert_16bit_to_8bit_c;
    pack2d_16_bit_src_mul4 = eb_enc_msb_pack2_d;
    un_pack2d_16_bit_src_mul4 = eb_enc_msb_un_pack2_d;

    full_distortion_kernel_cbf_zero32_bits = full_distortion_kernel_cbf_zero32_bits_c;
    full_distortion_kernel32_bits = full_distortion_kernel32_bits_c;

    spatial_full_distortion_kernel = spatial_full_distortion_kernel_c;
    full_distortion_kernel16_bits = full_distortion_kernel16_bits_c;
    residual_kernel8bit = residual_kernel8bit_c;

    residual_kernel16bit = residual_kernel16bit_c;

    picture_average_kernel = picture_average_kernel_c;
    picture_average_kernel1_line = picture_average_kernel1_line_c;

    avc_style_luma_interpolation_filter = avc_style_luma_interpolation_filter_helper_c;

    eb_av1_wiener_convolve_add_src = eb_av1_wiener_convolve_add_src_c,


    eb_av1_convolve_2d_copy_sr = eb_av1_convolve_2d_copy_sr_c;

    eb_av1_convolve_2d_scale = eb_av1_convolve_2d_scale_c;

    eb_av1_highbd_convolve_2d_copy_sr = eb_av1_highbd_convolve_2d_copy_sr_c;
    eb_av1_highbd_jnt_convolve_2d_copy = eb_av1_highbd_jnt_convolve_2d_copy_c;
    eb_av1_highbd_convolve_y_sr = eb_av1_highbd_convolve_y_sr_c;
    eb_av1_highbd_convolve_2d_sr = eb_av1_highbd_convolve_2d_sr_c;

    eb_av1_highbd_convolve_2d_scale = eb_av1_highbd_convolve_2d_scale_c;

    eb_av1_highbd_jnt_convolve_2d = eb_av1_highbd_jnt_convolve_2d_c;
    eb_av1_highbd_jnt_convolve_x = eb_av1_highbd_jnt_convolve_x_c;
    eb_av1_highbd_jnt_convolve_y = eb_av1_highbd_jnt_convolve_y_c;
    eb_av1_highbd_convolve_x_sr = eb_av1_highbd_convolve_x_sr_c;

    eb_av1_convolve_2d_sr = eb_av1_convolve_2d_sr_c;
    eb_av1_convolve_2d_copy_sr = eb_av1_convolve_2d_copy_sr_c;
    eb_av1_convolve_x_sr = eb_av1_convolve_x_sr_c;
    eb_av1_convolve_y_sr = eb_av1_convolve_y_sr_c;
    eb_av1_jnt_convolve_2d = eb_av1_jnt_convolve_2d_c;
    eb_av1_jnt_convolve_2d_copy = eb_av1_jnt_convolve_2d_copy_c;
    eb_av1_jnt_convolve_x = eb_av1_jnt_convolve_x_c;
    eb_av1_jnt_convolve_y = eb_av1_jnt_convolve_y_c;

    aom_convolve8_horiz = aom_convolve8_horiz_c;
    aom_convolve8_vert = aom_convolve8_vert_c;


    av1_build_compound_diffwtd_mask = av1_build_compound_diffwtd_mask_c;
    av1_build_compound_diffwtd_mask_highbd = av1_build_compound_diffwtd_mask_highbd_c;
    av1_wedge_sse_from_residuals = av1_wedge_sse_from_residuals_c;

    aom_subtract_block = aom_subtract_block_c;

    aom_lowbd_blend_a64_d16_mask = aom_lowbd_blend_a64_d16_mask_c;
    aom_highbd_blend_a64_d16_mask = aom_highbd_blend_a64_d16_mask_c;

    aom_highbd_subtract_block = aom_highbd_subtract_block_c;

    eb_aom_highbd_smooth_v_predictor_16x16 = eb_aom_highbd_smooth_v_predictor_16x16_c;
    eb_aom_highbd_smooth_v_predictor_16x32 = eb_aom_highbd_smooth_v_predictor_16x32_c;
    eb_aom_highbd_smooth_v_predictor_16x4 = eb_aom_highbd_smooth_v_predictor_16x4_c;
    eb_aom_highbd_smooth_v_predictor_16x64 = eb_aom_highbd_smooth_v_predictor_16x64_c;
    eb_aom_highbd_smooth_v_predictor_16x8 = eb_aom_highbd_smooth_v_predictor_16x8_c;
    eb_aom_highbd_smooth_v_predictor_2x2 = eb_aom_highbd_smooth_v_predictor_2x2_c;
    eb_aom_highbd_smooth_v_predictor_32x16 = eb_aom_highbd_smooth_v_predictor_32x16_c;
    eb_aom_highbd_smooth_v_predictor_32x32 = eb_aom_highbd_smooth_v_predictor_32x32_c;
    eb_aom_highbd_smooth_v_predictor_32x64 = eb_aom_highbd_smooth_v_predictor_32x64_c;
    eb_aom_highbd_smooth_v_predictor_32x8 = eb_aom_highbd_smooth_v_predictor_32x8_c;
    eb_aom_highbd_smooth_v_predictor_4x16 = eb_aom_highbd_smooth_v_predictor_4x16_c;
    eb_aom_highbd_smooth_v_predictor_4x4 = eb_aom_highbd_smooth_v_predictor_4x4_c;
    eb_aom_highbd_smooth_v_predictor_4x8 = eb_aom_highbd_smooth_v_predictor_4x8_c;
    eb_aom_highbd_smooth_v_predictor_64x16 = eb_aom_highbd_smooth_v_predictor_64x16_c;
    eb_aom_highbd_smooth_v_predictor_64x32 = eb_aom_highbd_smooth_v_predictor_64x32_c;
    eb_aom_highbd_smooth_v_predictor_64x64 = eb_aom_highbd_smooth_v_predictor_64x64_c;
    eb_aom_highbd_smooth_v_predictor_8x16 = eb_aom_highbd_smooth_v_predictor_8x16_c;
    eb_aom_highbd_smooth_v_predictor_8x32 = eb_aom_highbd_smooth_v_predictor_8x32_c;
    eb_aom_highbd_smooth_v_predictor_8x4 = eb_aom_highbd_smooth_v_predictor_8x4_c;
    eb_aom_highbd_smooth_v_predictor_8x8 = eb_aom_highbd_smooth_v_predictor_8x8_c;


    eb_av1_dr_prediction_z1 = eb_av1_dr_prediction_z1_c;
    eb_av1_dr_prediction_z2 = eb_av1_dr_prediction_z2_c;
    eb_av1_dr_prediction_z3 = eb_av1_dr_prediction_z3_c;
    eb_av1_highbd_dr_prediction_z1 = eb_av1_highbd_dr_prediction_z1_c;
    eb_av1_highbd_dr_prediction_z2 = eb_av1_highbd_dr_prediction_z2_c;
    eb_av1_highbd_dr_prediction_z3 = eb_av1_highbd_dr_prediction_z3_c;

    eb_aom_paeth_predictor_16x16 = eb_aom_paeth_predictor_16x16_c;
    eb_aom_paeth_predictor_16x32 = eb_aom_paeth_predictor_16x32_c;
    eb_aom_paeth_predictor_16x4 = eb_aom_paeth_predictor_16x4_c;
    eb_aom_paeth_predictor_16x64 = eb_aom_paeth_predictor_16x64_c;
    eb_aom_paeth_predictor_16x8 = eb_aom_paeth_predictor_16x8_c;
    eb_aom_paeth_predictor_32x16 = eb_aom_paeth_predictor_32x16_c;
    eb_aom_paeth_predictor_32x32 = eb_aom_paeth_predictor_32x32_c;
    eb_aom_paeth_predictor_32x64 = eb_aom_paeth_predictor_32x64_c;
    eb_aom_paeth_predictor_32x8 = eb_aom_paeth_predictor_32x8_c;
    eb_aom_paeth_predictor_4x16 = eb_aom_paeth_predictor_4x16_c;
    eb_aom_paeth_predictor_4x4 = eb_aom_paeth_predictor_4x4_c;
    eb_aom_paeth_predictor_4x8 = eb_aom_paeth_predictor_4x8_c;
    eb_aom_paeth_predictor_64x16 = eb_aom_paeth_predictor_64x16_c;
    eb_aom_paeth_predictor_64x32 = eb_aom_paeth_predictor_64x32_c;
    eb_aom_paeth_predictor_64x64 = eb_aom_paeth_predictor_64x64_c;
    eb_aom_paeth_predictor_8x16 = eb_aom_paeth_predictor_8x16_c;
    eb_aom_paeth_predictor_8x32 = eb_aom_paeth_predictor_8x32_c;
    eb_aom_paeth_predictor_8x4 = eb_aom_paeth_predictor_8x4_c;
    eb_aom_paeth_predictor_8x8 = eb_aom_paeth_predictor_8x8_c;

    eb_aom_highbd_paeth_predictor_16x16 = eb_aom_highbd_paeth_predictor_16x16_c;
    eb_aom_highbd_paeth_predictor_16x32 = eb_aom_highbd_paeth_predictor_16x32_c;
    eb_aom_highbd_paeth_predictor_16x4 = eb_aom_highbd_paeth_predictor_16x4_c;
    eb_aom_highbd_paeth_predictor_16x64 = eb_aom_highbd_paeth_predictor_16x64_c;
    eb_aom_highbd_paeth_predictor_16x8 = eb_aom_highbd_paeth_predictor_16x8_c;
    eb_aom_highbd_paeth_predictor_2x2 = eb_aom_highbd_paeth_predictor_2x2_c;
    eb_aom_highbd_paeth_predictor_32x16 = eb_aom_highbd_paeth_predictor_32x16_c;
    eb_aom_highbd_paeth_predictor_32x32 = eb_aom_highbd_paeth_predictor_32x32_c;
    eb_aom_highbd_paeth_predictor_32x64 = eb_aom_highbd_paeth_predictor_32x64_c;
    eb_aom_highbd_paeth_predictor_32x8 = eb_aom_highbd_paeth_predictor_32x8_c;
    eb_aom_highbd_paeth_predictor_4x16 = eb_aom_highbd_paeth_predictor_4x16_c;
    eb_aom_highbd_paeth_predictor_4x4 = eb_aom_highbd_paeth_predictor_4x4_c;
    eb_aom_highbd_paeth_predictor_4x8 = eb_aom_highbd_paeth_predictor_4x8_c;
    eb_aom_highbd_paeth_predictor_64x16 = eb_aom_highbd_paeth_predictor_64x16_c;
    eb_aom_highbd_paeth_predictor_64x32 = eb_aom_highbd_paeth_predictor_64x32_c;
    eb_aom_highbd_paeth_predictor_64x64 = eb_aom_highbd_paeth_predictor_64x64_c;
    eb_aom_highbd_paeth_predictor_8x16 = eb_aom_highbd_paeth_predictor_8x16_c;
    eb_aom_highbd_paeth_predictor_8x32 = eb_aom_highbd_paeth_predictor_8x32_c;
    eb_aom_highbd_paeth_predictor_8x4 = eb_aom_highbd_paeth_predictor_8x4_c;
    eb_aom_highbd_paeth_predictor_8x8 = eb_aom_highbd_paeth_predictor_8x8_c;
    aom_sum_squares_i16 = aom_sum_squares_i16_c;
    eb_aom_dc_predictor_4x4 = eb_aom_dc_predictor_4x4_c;
    eb_aom_dc_predictor_8x8 = eb_aom_dc_predictor_8x8_c;
    eb_aom_dc_predictor_16x16 = eb_aom_dc_predictor_16x16_c;
    eb_aom_dc_predictor_32x32 = eb_aom_dc_predictor_32x32_c;
    eb_aom_dc_predictor_64x64 = eb_aom_dc_predictor_64x64_c;
    eb_aom_dc_predictor_32x16 = eb_aom_dc_predictor_32x16_c;
    eb_aom_dc_predictor_32x64 = eb_aom_dc_predictor_32x64_c;
    eb_aom_dc_predictor_64x16 = eb_aom_dc_predictor_64x16_c;
    eb_aom_dc_predictor_8x16 = eb_aom_dc_predictor_8x16_c;
    eb_aom_dc_predictor_8x32 = eb_aom_dc_predictor_8x32_c;
    eb_aom_dc_predictor_8x4 = eb_aom_dc_predictor_8x4_c;
    eb_aom_dc_predictor_64x32 = eb_aom_dc_predictor_64x32_c;
    eb_aom_dc_predictor_16x32 = eb_aom_dc_predictor_16x32_c;
    eb_aom_dc_predictor_16x4 = eb_aom_dc_predictor_16x4_c;
    eb_aom_dc_predictor_16x64 = eb_aom_dc_predictor_16x64_c;
    eb_aom_dc_predictor_16x8 = eb_aom_dc_predictor_16x8_c;
    eb_aom_dc_predictor_32x8 = eb_aom_dc_predictor_32x8_c;
    eb_aom_dc_predictor_4x16 = eb_aom_dc_predictor_4x16_c;
    eb_aom_dc_predictor_4x8 = eb_aom_dc_predictor_4x8_c;

    eb_aom_dc_top_predictor_4x4 = eb_aom_dc_top_predictor_4x4_c;
    eb_aom_dc_top_predictor_8x8 = eb_aom_dc_top_predictor_8x8_c;
    eb_aom_dc_top_predictor_16x16 = eb_aom_dc_top_predictor_16x16_c;
    eb_aom_dc_top_predictor_32x32 = eb_aom_dc_top_predictor_32x32_c;
    eb_aom_dc_top_predictor_64x64 = eb_aom_dc_top_predictor_64x64_c;
    eb_aom_dc_top_predictor_16x32 = eb_aom_dc_top_predictor_16x32_c;
    eb_aom_dc_top_predictor_16x4 = eb_aom_dc_top_predictor_16x4_c;
    eb_aom_dc_top_predictor_16x64 = eb_aom_dc_top_predictor_16x64_c;
    eb_aom_dc_top_predictor_16x8 = eb_aom_dc_top_predictor_16x8_c;
    eb_aom_dc_top_predictor_32x16 = eb_aom_dc_top_predictor_32x16_c;
    eb_aom_dc_top_predictor_32x64 = eb_aom_dc_top_predictor_32x64_c;
    eb_aom_dc_top_predictor_32x8 = eb_aom_dc_top_predictor_32x8_c;
    eb_aom_dc_top_predictor_4x16 = eb_aom_dc_top_predictor_4x16_c;
    eb_aom_dc_top_predictor_4x8 = eb_aom_dc_top_predictor_4x8_c;
    eb_aom_dc_top_predictor_64x16 = eb_aom_dc_top_predictor_64x16_c;
    eb_aom_dc_top_predictor_64x32 = eb_aom_dc_top_predictor_64x32_c;
    eb_aom_dc_top_predictor_8x16 = eb_aom_dc_top_predictor_8x16_c;
    eb_aom_dc_top_predictor_8x32 = eb_aom_dc_top_predictor_8x32_c;
    eb_aom_dc_top_predictor_8x4 = eb_aom_dc_top_predictor_8x4_c;

    eb_aom_dc_left_predictor_4x4 = eb_aom_dc_left_predictor_4x4_c;
    eb_aom_dc_left_predictor_8x8 = eb_aom_dc_left_predictor_8x8_c;
    eb_aom_dc_left_predictor_16x16 = eb_aom_dc_left_predictor_16x16_c;
    eb_aom_dc_left_predictor_32x32 = eb_aom_dc_left_predictor_32x32_c;
    eb_aom_dc_left_predictor_64x64 = eb_aom_dc_left_predictor_64x64_c;
    eb_aom_dc_left_predictor_16x32 = eb_aom_dc_left_predictor_16x32_c;
    eb_aom_dc_left_predictor_16x4 = eb_aom_dc_left_predictor_16x4_c;
    eb_aom_dc_left_predictor_16x64 = eb_aom_dc_left_predictor_16x64_c;
    eb_aom_dc_left_predictor_16x8 = eb_aom_dc_left_predictor_16x8_c;
    eb_aom_dc_left_predictor_32x16 = eb_aom_dc_left_predictor_32x16_c;
    eb_aom_dc_left_predictor_32x64 = eb_aom_dc_left_predictor_32x64_c;
    eb_aom_dc_left_predictor_64x16 = eb_aom_dc_left_predictor_64x16_c;
    eb_aom_dc_left_predictor_64x32 = eb_aom_dc_left_predictor_64x32_c;
    eb_aom_dc_left_predictor_32x8 = eb_aom_dc_left_predictor_32x8_c;
    eb_aom_dc_left_predictor_4x16 = eb_aom_dc_left_predictor_4x16_c;
    eb_aom_dc_left_predictor_4x8 = eb_aom_dc_left_predictor_4x8_c;
    eb_aom_dc_left_predictor_8x16 = eb_aom_dc_left_predictor_8x16_c;
    eb_aom_dc_left_predictor_8x32 = eb_aom_dc_left_predictor_8x32_c;
    eb_aom_dc_left_predictor_8x4 = eb_aom_dc_left_predictor_8x4_c;

    eb_aom_dc_128_predictor_4x4 = eb_aom_dc_128_predictor_4x4_c;
    eb_aom_dc_128_predictor_8x8 = eb_aom_dc_128_predictor_8x8_c;
    eb_aom_dc_128_predictor_16x16 = eb_aom_dc_128_predictor_16x16_c;
    eb_aom_dc_128_predictor_32x32 = eb_aom_dc_128_predictor_32x32_c;
    eb_aom_dc_128_predictor_64x64 = eb_aom_dc_128_predictor_64x64_c;
    eb_aom_dc_128_predictor_16x32 = eb_aom_dc_128_predictor_16x32_c;
    eb_aom_dc_128_predictor_16x4 = eb_aom_dc_128_predictor_16x4_c;
    eb_aom_dc_128_predictor_16x64 = eb_aom_dc_128_predictor_16x64_c;
    eb_aom_dc_128_predictor_16x8 = eb_aom_dc_128_predictor_16x8_c;
    eb_aom_dc_128_predictor_32x16 = eb_aom_dc_128_predictor_32x16_c;
    eb_aom_dc_128_predictor_32x64 = eb_aom_dc_128_predictor_32x64_c;
    eb_aom_dc_128_predictor_32x8 = eb_aom_dc_128_predictor_32x8_c;
    eb_aom_dc_128_predictor_4x16 = eb_aom_dc_128_predictor_4x16_c;
    eb_aom_dc_128_predictor_4x8 = eb_aom_dc_128_predictor_4x8_c;
    eb_aom_dc_128_predictor_64x16 = eb_aom_dc_128_predictor_64x16_c;
    eb_aom_dc_128_predictor_64x32 = eb_aom_dc_128_predictor_64x32_c;
    eb_aom_dc_128_predictor_8x16 = eb_aom_dc_128_predictor_8x16_c;
    eb_aom_dc_128_predictor_8x32 = eb_aom_dc_128_predictor_8x32_c;
    eb_aom_dc_128_predictor_8x4 = eb_aom_dc_128_predictor_8x4_c;

    eb_aom_smooth_h_predictor_16x32 = eb_aom_smooth_h_predictor_16x32_c;
    eb_aom_smooth_h_predictor_16x4 = eb_aom_smooth_h_predictor_16x4_c;
    eb_aom_smooth_h_predictor_16x64 = eb_aom_smooth_h_predictor_16x64_c;
    eb_aom_smooth_h_predictor_16x8 = eb_aom_smooth_h_predictor_16x8_c;
    eb_aom_smooth_h_predictor_32x16 = eb_aom_smooth_h_predictor_32x16_c;
    eb_aom_smooth_h_predictor_32x64 = eb_aom_smooth_h_predictor_32x64_c;
    eb_aom_smooth_h_predictor_32x8 = eb_aom_smooth_h_predictor_32x8_c;
    eb_aom_smooth_h_predictor_4x16 = eb_aom_smooth_h_predictor_4x16_c;
    eb_aom_smooth_h_predictor_4x8 = eb_aom_smooth_h_predictor_4x8_c;
    eb_aom_smooth_h_predictor_64x16 = eb_aom_smooth_h_predictor_64x16_c;
    eb_aom_smooth_h_predictor_64x32 = eb_aom_smooth_h_predictor_64x32_c;
    eb_aom_smooth_h_predictor_8x16 = eb_aom_smooth_h_predictor_8x16_c;
    eb_aom_smooth_h_predictor_8x32 = eb_aom_smooth_h_predictor_8x32_c;
    eb_aom_smooth_h_predictor_8x4 = eb_aom_smooth_h_predictor_8x4_c;
    eb_aom_smooth_h_predictor_64x64 = eb_aom_smooth_h_predictor_64x64_c;
    eb_aom_smooth_h_predictor_32x32 = eb_aom_smooth_h_predictor_32x32_c;
    eb_aom_smooth_h_predictor_16x16 = eb_aom_smooth_h_predictor_16x16_c;
    eb_aom_smooth_h_predictor_8x8 = eb_aom_smooth_h_predictor_8x8_c;
    eb_aom_smooth_h_predictor_4x4 = eb_aom_smooth_h_predictor_4x4_c;
    eb_aom_smooth_v_predictor_16x32 = eb_aom_smooth_v_predictor_16x32_c;
    eb_aom_smooth_v_predictor_16x4 = eb_aom_smooth_v_predictor_16x4_c;
    eb_aom_smooth_v_predictor_16x64 = eb_aom_smooth_v_predictor_16x64_c;
    eb_aom_smooth_v_predictor_16x8 = eb_aom_smooth_v_predictor_16x8_c;
    eb_aom_smooth_v_predictor_32x16 = eb_aom_smooth_v_predictor_32x16_c;
    eb_aom_smooth_v_predictor_32x64 = eb_aom_smooth_v_predictor_32x64_c;
    eb_aom_smooth_v_predictor_32x8 = eb_aom_smooth_v_predictor_32x8_c;
    eb_aom_smooth_v_predictor_4x16 = eb_aom_smooth_v_predictor_4x16_c;
    eb_aom_smooth_v_predictor_4x8 = eb_aom_smooth_v_predictor_4x8_c;
    eb_aom_smooth_v_predictor_64x16 = eb_aom_smooth_v_predictor_64x16_c;
    eb_aom_smooth_v_predictor_64x32 = eb_aom_smooth_v_predictor_64x32_c;
    eb_aom_smooth_v_predictor_8x16 = eb_aom_smooth_v_predictor_8x16_c;
    eb_aom_smooth_v_predictor_8x32 = eb_aom_smooth_v_predictor_8x32_c;
    eb_aom_smooth_v_predictor_8x4 = eb_aom_smooth_v_predictor_8x4_c;
    eb_aom_smooth_v_predictor_64x64 = eb_aom_smooth_v_predictor_64x64_c;
    eb_aom_smooth_v_predictor_32x32 = eb_aom_smooth_v_predictor_32x32_c;
    eb_aom_smooth_v_predictor_16x16 = eb_aom_smooth_v_predictor_16x16_c;
    eb_aom_smooth_v_predictor_8x8 = eb_aom_smooth_v_predictor_8x8_c;
    eb_aom_smooth_v_predictor_4x4 = eb_aom_smooth_v_predictor_4x4_c;

    eb_aom_smooth_predictor_16x32 = eb_aom_smooth_predictor_16x32_c;
    eb_aom_smooth_predictor_16x4 = eb_aom_smooth_predictor_16x4_c;
    eb_aom_smooth_predictor_16x64 = eb_aom_smooth_predictor_16x64_c;
    eb_aom_smooth_predictor_16x8 = eb_aom_smooth_predictor_16x8_c;
    eb_aom_smooth_predictor_32x16 = eb_aom_smooth_predictor_32x16_c;
    eb_aom_smooth_predictor_32x64 = eb_aom_smooth_predictor_32x64_c;
    eb_aom_smooth_predictor_32x8 = eb_aom_smooth_predictor_32x8_c;
    eb_aom_smooth_predictor_4x16 = eb_aom_smooth_predictor_4x16_c;
    eb_aom_smooth_predictor_4x8 = eb_aom_smooth_predictor_4x8_c;
    eb_aom_smooth_predictor_64x16 = eb_aom_smooth_predictor_64x16_c;
    eb_aom_smooth_predictor_64x32 = eb_aom_smooth_predictor_64x32_c;
    eb_aom_smooth_predictor_8x16 = eb_aom_smooth_predictor_8x16_c;
    eb_aom_smooth_predictor_8x32 = eb_aom_smooth_predictor_8x32_c;
    eb_aom_smooth_predictor_8x4 = eb_aom_smooth_predictor_8x4_c;
    eb_aom_smooth_predictor_64x64 = eb_aom_smooth_predictor_64x64_c;
    eb_aom_smooth_predictor_32x32 = eb_aom_smooth_predictor_32x32_c;
    eb_aom_smooth_predictor_16x16 = eb_aom_smooth_predictor_16x16_c;
    eb_aom_smooth_predictor_8x8 = eb_aom_smooth_predictor_8x8_c;
    eb_aom_smooth_predictor_4x4 = eb_aom_smooth_predictor_4x4_c;

    eb_aom_v_predictor_4x4 = eb_aom_v_predictor_4x4_c;
    eb_aom_v_predictor_8x8 = eb_aom_v_predictor_8x8_c;
    eb_aom_v_predictor_16x16 = eb_aom_v_predictor_16x16_c;
    eb_aom_v_predictor_32x32 = eb_aom_v_predictor_32x32_c;
    eb_aom_v_predictor_64x64 = eb_aom_v_predictor_64x64_c;
    eb_aom_v_predictor_16x32 = eb_aom_v_predictor_16x32_c;
    eb_aom_v_predictor_16x4 = eb_aom_v_predictor_16x4_c;
    eb_aom_v_predictor_16x64 = eb_aom_v_predictor_16x64_c;
    eb_aom_v_predictor_16x8 = eb_aom_v_predictor_16x8_c;
    eb_aom_v_predictor_32x16 = eb_aom_v_predictor_32x16_c;
    eb_aom_v_predictor_32x64 = eb_aom_v_predictor_32x64_c;
    eb_aom_v_predictor_32x8 = eb_aom_v_predictor_32x8_c;
    eb_aom_v_predictor_4x16 = eb_aom_v_predictor_4x16_c;
    eb_aom_v_predictor_4x8 = eb_aom_v_predictor_4x8_c;
    eb_aom_v_predictor_64x16 = eb_aom_v_predictor_64x16_c;
    eb_aom_v_predictor_64x32 = eb_aom_v_predictor_64x32_c;
    eb_aom_v_predictor_8x16 = eb_aom_v_predictor_8x16_c;
    eb_aom_v_predictor_8x32 = eb_aom_v_predictor_8x32_c;
    eb_aom_v_predictor_8x4 = eb_aom_v_predictor_8x4_c;

    eb_aom_h_predictor_4x4 = eb_aom_h_predictor_4x4_c;
    eb_aom_h_predictor_8x8 = eb_aom_h_predictor_8x8_c;
    eb_aom_h_predictor_16x16 = eb_aom_h_predictor_16x16_c;
    eb_aom_h_predictor_32x32 = eb_aom_h_predictor_32x32_c;
    eb_aom_h_predictor_64x64 = eb_aom_h_predictor_64x64_c;
    eb_aom_h_predictor_16x32 = eb_aom_h_predictor_16x32_c;
    eb_aom_h_predictor_16x4 = eb_aom_h_predictor_16x4_c;
    eb_aom_h_predictor_16x64 = eb_aom_h_predictor_16x64_c;
    eb_aom_h_predictor_16x8 = eb_aom_h_predictor_16x8_c;
    eb_aom_h_predictor_32x16 = eb_aom_h_predictor_32x16_c;
    eb_aom_h_predictor_32x64 = eb_aom_h_predictor_32x64_c;
    eb_aom_h_predictor_32x8 = eb_aom_h_predictor_32x8_c;
    eb_aom_h_predictor_4x16 = eb_aom_h_predictor_4x16_c;
    eb_aom_h_predictor_4x8 = eb_aom_h_predictor_4x8_c;
    eb_aom_h_predictor_64x16 = eb_aom_h_predictor_64x16_c;
    eb_aom_h_predictor_64x32 = eb_aom_h_predictor_64x32_c;
    eb_aom_h_predictor_8x16 = eb_aom_h_predictor_8x16_c;
    eb_aom_h_predictor_8x32 = eb_aom_h_predictor_8x32_c;
    eb_aom_h_predictor_8x4 = eb_aom_h_predictor_8x4_c;
    aom_sum_squares_i16 = aom_sum_squares_i16_c;
    eb_cdef_find_dir = eb_cdef_find_dir_c;

    eb_cdef_filter_block = eb_cdef_filter_block_c;

    eb_copy_rect8_8bit_to_16bit = eb_copy_rect8_8bit_to_16bit_c;


    eb_av1_highbd_warp_affine = eb_av1_highbd_warp_affine_c;

    eb_av1_warp_affine = eb_av1_warp_affine_c;

    aom_highbd_lpf_horizontal_14 = aom_highbd_lpf_horizontal_14_c;
    aom_highbd_lpf_horizontal_4 = aom_highbd_lpf_horizontal_4_c;
    aom_highbd_lpf_horizontal_6 = aom_highbd_lpf_horizontal_6_c;
    aom_highbd_lpf_horizontal_8 = aom_highbd_lpf_horizontal_8_c;
    aom_highbd_lpf_vertical_14 = aom_highbd_lpf_vertical_14_c;
    aom_highbd_lpf_vertical_4 = aom_highbd_lpf_vertical_4_c;
    aom_highbd_lpf_vertical_6 = aom_highbd_lpf_vertical_6_c;
    aom_highbd_lpf_vertical_8 = aom_highbd_lpf_vertical_8_c;
    aom_lpf_horizontal_14 = aom_lpf_horizontal_14_c;
    aom_lpf_horizontal_4 = aom_lpf_horizontal_4_c;
    aom_lpf_horizontal_6 = aom_lpf_horizontal_6_c;
    aom_lpf_horizontal_8 = aom_lpf_horizontal_8_c;
    aom_lpf_vertical_14 = aom_lpf_vertical_14_c;
    aom_lpf_vertical_4 = aom_lpf_vertical_4_c;
    aom_lpf_vertical_6 = aom_lpf_vertical_6_c;
    aom_lpf_vertical_8 = aom_lpf_vertical_8_c;

    // eb_aom_highbd_v_predictor
    eb_aom_highbd_v_predictor_16x16 = eb_aom_highbd_v_predictor_16x16_c;
    eb_aom_highbd_v_predictor_16x32 = eb_aom_highbd_v_predictor_16x32_c;
    eb_aom_highbd_v_predictor_16x4 = eb_aom_highbd_v_predictor_16x4_c;
    eb_aom_highbd_v_predictor_16x64 = eb_aom_highbd_v_predictor_16x64_c;
    eb_aom_highbd_v_predictor_16x8 = eb_aom_highbd_v_predictor_16x8_c;
    eb_aom_highbd_v_predictor_32x16 = eb_aom_highbd_v_predictor_32x16_c;
    eb_aom_highbd_v_predictor_32x32 = eb_aom_highbd_v_predictor_32x32_c;
    eb_aom_highbd_v_predictor_32x64 = eb_aom_highbd_v_predictor_32x64_c;
    eb_aom_highbd_v_predictor_32x8 = eb_aom_highbd_v_predictor_32x8_c;
    eb_aom_highbd_v_predictor_4x16 = eb_aom_highbd_v_predictor_4x16_c;
    eb_aom_highbd_v_predictor_4x4 = eb_aom_highbd_v_predictor_4x4_c;
    eb_aom_highbd_v_predictor_4x8 = eb_aom_highbd_v_predictor_4x8_c;
    eb_aom_highbd_v_predictor_64x16 = eb_aom_highbd_v_predictor_64x16_c;
    eb_aom_highbd_v_predictor_64x32 = eb_aom_highbd_v_predictor_64x32_c;
    eb_aom_highbd_v_predictor_8x32 = eb_aom_highbd_v_predictor_8x32_c;
    eb_aom_highbd_v_predictor_64x64 = eb_aom_highbd_v_predictor_64x64_c;
    eb_aom_highbd_v_predictor_8x16 = eb_aom_highbd_v_predictor_8x16_c;
    eb_aom_highbd_v_predictor_8x4 = eb_aom_highbd_v_predictor_8x4_c;
    eb_aom_highbd_v_predictor_8x8 = eb_aom_highbd_v_predictor_8x8_c;

    //aom_highbd_smooth_predictor
    eb_aom_highbd_smooth_predictor_16x16 = eb_aom_highbd_smooth_predictor_16x16_c;
    eb_aom_highbd_smooth_predictor_16x32 = eb_aom_highbd_smooth_predictor_16x32_c;
    eb_aom_highbd_smooth_predictor_16x4 = eb_aom_highbd_smooth_predictor_16x4_c;
    eb_aom_highbd_smooth_predictor_16x64 = eb_aom_highbd_smooth_predictor_16x64_c;
    eb_aom_highbd_smooth_predictor_16x8 = eb_aom_highbd_smooth_predictor_16x8_c;
    eb_aom_highbd_smooth_predictor_2x2 = eb_aom_highbd_smooth_predictor_2x2_c;
    eb_aom_highbd_smooth_predictor_32x16 = eb_aom_highbd_smooth_predictor_32x16_c;
    eb_aom_highbd_smooth_predictor_32x32 = eb_aom_highbd_smooth_predictor_32x32_c;
    eb_aom_highbd_smooth_predictor_32x64 = eb_aom_highbd_smooth_predictor_32x64_c;
    eb_aom_highbd_smooth_predictor_32x8 = eb_aom_highbd_smooth_predictor_32x8_c;
    eb_aom_highbd_smooth_predictor_4x16 = eb_aom_highbd_smooth_predictor_4x16_c;
    eb_aom_highbd_smooth_predictor_4x4 = eb_aom_highbd_smooth_predictor_4x4_c;
    eb_aom_highbd_smooth_predictor_4x8 = eb_aom_highbd_smooth_predictor_4x8_c;
    eb_aom_highbd_smooth_predictor_64x16 = eb_aom_highbd_smooth_predictor_64x16_c;
    eb_aom_highbd_smooth_predictor_64x32 = eb_aom_highbd_smooth_predictor_64x32_c;
    eb_aom_highbd_smooth_predictor_64x64 = eb_aom_highbd_smooth_predictor_64x64_c;
    eb_aom_highbd_smooth_predictor_8x16 = eb_aom_highbd_smooth_predictor_8x16_c;
    eb_aom_highbd_smooth_predictor_8x32 = eb_aom_highbd_smooth_predictor_8x32_c;
    eb_aom_highbd_smooth_predictor_8x4 = eb_aom_highbd_smooth_predictor_8x4_c;
    eb_aom_highbd_smooth_predictor_8x8 = eb_aom_highbd_smooth_predictor_8x8_c;


    //aom_highbd_smooth_h_predictor
    eb_aom_highbd_smooth_h_predictor_16x16 = eb_aom_highbd_smooth_h_predictor_16x16_c;
    eb_aom_highbd_smooth_h_predictor_16x32 = eb_aom_highbd_smooth_h_predictor_16x32_c;
    eb_aom_highbd_smooth_h_predictor_16x4 = eb_aom_highbd_smooth_h_predictor_16x4_c;
    eb_aom_highbd_smooth_h_predictor_16x64 = eb_aom_highbd_smooth_h_predictor_16x64_c;
    eb_aom_highbd_smooth_h_predictor_16x8 = eb_aom_highbd_smooth_h_predictor_16x8_c;
    eb_aom_highbd_smooth_h_predictor_32x16 = eb_aom_highbd_smooth_h_predictor_32x16_c;
    eb_aom_highbd_smooth_h_predictor_32x32 = eb_aom_highbd_smooth_h_predictor_32x32_c;
    eb_aom_highbd_smooth_h_predictor_32x64 = eb_aom_highbd_smooth_h_predictor_32x64_c;
    eb_aom_highbd_smooth_h_predictor_32x8 = eb_aom_highbd_smooth_h_predictor_32x8_c;
    eb_aom_highbd_smooth_h_predictor_4x16 = eb_aom_highbd_smooth_h_predictor_4x16_c;
    eb_aom_highbd_smooth_h_predictor_4x4 = eb_aom_highbd_smooth_h_predictor_4x4_c;
    eb_aom_highbd_smooth_h_predictor_4x8 = eb_aom_highbd_smooth_h_predictor_4x8_c;
    eb_aom_highbd_smooth_h_predictor_64x16 = eb_aom_highbd_smooth_h_predictor_64x16_c;
    eb_aom_highbd_smooth_h_predictor_64x32 = eb_aom_highbd_smooth_h_predictor_64x32_c;
    eb_aom_highbd_smooth_h_predictor_64x64 = eb_aom_highbd_smooth_h_predictor_64x64_c;
    eb_aom_highbd_smooth_h_predictor_8x16 = eb_aom_highbd_smooth_h_predictor_8x16_c;
    eb_aom_highbd_smooth_h_predictor_8x32 = eb_aom_highbd_smooth_h_predictor_8x32_c;
    eb_aom_highbd_smooth_h_predictor_8x4 = eb_aom_highbd_smooth_h_predictor_8x4_c;
    eb_aom_highbd_smooth_h_predictor_8x8 = eb_aom_highbd_smooth_h_predictor_8x8_c;

    //aom_highbd_dc_128_predictor
    eb_aom_highbd_dc_128_predictor_16x16 = eb_aom_highbd_dc_128_predictor_16x16_c;
    eb_aom_highbd_dc_128_predictor_16x32 = eb_aom_highbd_dc_128_predictor_16x32_c;
    eb_aom_highbd_dc_128_predictor_16x4 = eb_aom_highbd_dc_128_predictor_16x4_c;
    eb_aom_highbd_dc_128_predictor_16x64 = eb_aom_highbd_dc_128_predictor_16x64_c;
    eb_aom_highbd_dc_128_predictor_16x8 = eb_aom_highbd_dc_128_predictor_16x8_c;
    eb_aom_highbd_dc_128_predictor_32x16 = eb_aom_highbd_dc_128_predictor_32x16_c;
    eb_aom_highbd_dc_128_predictor_32x32 = eb_aom_highbd_dc_128_predictor_32x32_c;
    eb_aom_highbd_dc_128_predictor_32x64 = eb_aom_highbd_dc_128_predictor_32x64_c;
    eb_aom_highbd_dc_128_predictor_32x8 = eb_aom_highbd_dc_128_predictor_32x8_c;
    eb_aom_highbd_dc_128_predictor_4x16 = eb_aom_highbd_dc_128_predictor_4x16_c;
    eb_aom_highbd_dc_128_predictor_4x4 = eb_aom_highbd_dc_128_predictor_4x4_c;
    eb_aom_highbd_dc_128_predictor_4x8 = eb_aom_highbd_dc_128_predictor_4x8_c;
    eb_aom_highbd_dc_128_predictor_8x32 = eb_aom_highbd_dc_128_predictor_8x32_c;
    eb_aom_highbd_dc_128_predictor_64x16 = eb_aom_highbd_dc_128_predictor_64x16_c;
    eb_aom_highbd_dc_128_predictor_64x32 = eb_aom_highbd_dc_128_predictor_64x32_c;
    eb_aom_highbd_dc_128_predictor_64x64 = eb_aom_highbd_dc_128_predictor_64x64_c;
    eb_aom_highbd_dc_128_predictor_8x16 = eb_aom_highbd_dc_128_predictor_8x16_c;
    eb_aom_highbd_dc_128_predictor_8x4 = eb_aom_highbd_dc_128_predictor_8x4_c;
    eb_aom_highbd_dc_128_predictor_8x8 = eb_aom_highbd_dc_128_predictor_8x8_c;

    //aom_highbd_dc_left_predictor
    eb_aom_highbd_dc_left_predictor_16x16 = eb_aom_highbd_dc_left_predictor_16x16_c;
    eb_aom_highbd_dc_left_predictor_16x32 = eb_aom_highbd_dc_left_predictor_16x32_c;
    eb_aom_highbd_dc_left_predictor_16x4 = eb_aom_highbd_dc_left_predictor_16x4_c;
    eb_aom_highbd_dc_left_predictor_16x64 = eb_aom_highbd_dc_left_predictor_16x64_c;
    eb_aom_highbd_dc_left_predictor_16x8 = eb_aom_highbd_dc_left_predictor_16x8_c;
    eb_aom_highbd_dc_left_predictor_2x2 = eb_aom_highbd_dc_left_predictor_2x2_c;
    eb_aom_highbd_dc_left_predictor_32x16 = eb_aom_highbd_dc_left_predictor_32x16_c;
    eb_aom_highbd_dc_left_predictor_32x32 = eb_aom_highbd_dc_left_predictor_32x32_c;
    eb_aom_highbd_dc_left_predictor_32x64 = eb_aom_highbd_dc_left_predictor_32x64_c;
    eb_aom_highbd_dc_left_predictor_32x8 = eb_aom_highbd_dc_left_predictor_32x8_c;
    eb_aom_highbd_dc_left_predictor_4x16 = eb_aom_highbd_dc_left_predictor_4x16_c;
    eb_aom_highbd_dc_left_predictor_4x4 = eb_aom_highbd_dc_left_predictor_4x4_c;
    eb_aom_highbd_dc_left_predictor_4x8 = eb_aom_highbd_dc_left_predictor_4x8_c;
    eb_aom_highbd_dc_left_predictor_8x32 = eb_aom_highbd_dc_left_predictor_8x32_c;
    eb_aom_highbd_dc_left_predictor_64x16 = eb_aom_highbd_dc_left_predictor_64x16_c;
    eb_aom_highbd_dc_left_predictor_64x32 = eb_aom_highbd_dc_left_predictor_64x32_c;
    eb_aom_highbd_dc_left_predictor_64x64 = eb_aom_highbd_dc_left_predictor_64x64_c;
    eb_aom_highbd_dc_left_predictor_8x16 = eb_aom_highbd_dc_left_predictor_8x16_c;
    eb_aom_highbd_dc_left_predictor_8x4 = eb_aom_highbd_dc_left_predictor_8x4_c;
    eb_aom_highbd_dc_left_predictor_8x8 = eb_aom_highbd_dc_left_predictor_8x8_c;

    eb_aom_highbd_dc_predictor_16x16 = eb_aom_highbd_dc_predictor_16x16_c;
    eb_aom_highbd_dc_predictor_16x32 = eb_aom_highbd_dc_predictor_16x32_c;
    eb_aom_highbd_dc_predictor_16x4 = eb_aom_highbd_dc_predictor_16x4_c;
    eb_aom_highbd_dc_predictor_16x64 = eb_aom_highbd_dc_predictor_16x64_c;
    eb_aom_highbd_dc_predictor_16x8 = eb_aom_highbd_dc_predictor_16x8_c;
    eb_aom_highbd_dc_predictor_2x2 = eb_aom_highbd_dc_predictor_2x2_c;
    eb_aom_highbd_dc_predictor_32x16 = eb_aom_highbd_dc_predictor_32x16_c;
    eb_aom_highbd_dc_predictor_32x32 = eb_aom_highbd_dc_predictor_32x32_c;
    eb_aom_highbd_dc_predictor_32x64 = eb_aom_highbd_dc_predictor_32x64_c;
    eb_aom_highbd_dc_predictor_32x8 = eb_aom_highbd_dc_predictor_32x8_c;
    eb_aom_highbd_dc_predictor_4x16 = eb_aom_highbd_dc_predictor_4x16_c;
    eb_aom_highbd_dc_predictor_4x4 = eb_aom_highbd_dc_predictor_4x4_c;
    eb_aom_highbd_dc_predictor_4x8 = eb_aom_highbd_dc_predictor_4x8_c;
    eb_aom_highbd_dc_predictor_64x16 = eb_aom_highbd_dc_predictor_64x16_c;
    eb_aom_highbd_dc_predictor_64x32 = eb_aom_highbd_dc_predictor_64x32_c;
    eb_aom_highbd_dc_predictor_64x64 = eb_aom_highbd_dc_predictor_64x64_c;
    eb_aom_highbd_dc_predictor_8x16 = eb_aom_highbd_dc_predictor_8x16_c;
    eb_aom_highbd_dc_predictor_8x4 = eb_aom_highbd_dc_predictor_8x4_c;
    eb_aom_highbd_dc_predictor_8x8 = eb_aom_highbd_dc_predictor_8x8_c;
    eb_aom_highbd_dc_predictor_8x32 = eb_aom_highbd_dc_predictor_8x32_c;

    //aom_highbd_dc_top_predictor
    eb_aom_highbd_dc_top_predictor_16x16 = eb_aom_highbd_dc_top_predictor_16x16_c;
    eb_aom_highbd_dc_top_predictor_16x32 = eb_aom_highbd_dc_top_predictor_16x32_c;
    eb_aom_highbd_dc_top_predictor_16x4 = eb_aom_highbd_dc_top_predictor_16x4_c;
    eb_aom_highbd_dc_top_predictor_16x64 = eb_aom_highbd_dc_top_predictor_16x64_c;
    eb_aom_highbd_dc_top_predictor_16x8 = eb_aom_highbd_dc_top_predictor_16x8_c;
    eb_aom_highbd_dc_top_predictor_32x16 = eb_aom_highbd_dc_top_predictor_32x16_c;
    eb_aom_highbd_dc_top_predictor_32x32 = eb_aom_highbd_dc_top_predictor_32x32_c;
    eb_aom_highbd_dc_top_predictor_32x64 = eb_aom_highbd_dc_top_predictor_32x64_c;
    eb_aom_highbd_dc_top_predictor_32x8 = eb_aom_highbd_dc_top_predictor_32x8_c;
    eb_aom_highbd_dc_top_predictor_4x16 = eb_aom_highbd_dc_top_predictor_4x16_c;
    eb_aom_highbd_dc_top_predictor_4x4 = eb_aom_highbd_dc_top_predictor_4x4_c;
    eb_aom_highbd_dc_top_predictor_4x8 = eb_aom_highbd_dc_top_predictor_4x8_c;
    eb_aom_highbd_dc_top_predictor_64x16 = eb_aom_highbd_dc_top_predictor_64x16_c;
    eb_aom_highbd_dc_top_predictor_64x32 = eb_aom_highbd_dc_top_predictor_64x32_c;
    eb_aom_highbd_dc_top_predictor_64x64 = eb_aom_highbd_dc_top_predictor_64x64_c;
    eb_aom_highbd_dc_top_predictor_8x16 = eb_aom_highbd_dc_top_predictor_8x16_c;
    eb_aom_highbd_dc_top_predictor_8x32 = eb_aom_highbd_dc_top_predictor_8x32_c;
    eb_aom_highbd_dc_top_predictor_8x4 = eb_aom_highbd_dc_top_predictor_8x4_c;
    eb_aom_highbd_dc_top_predictor_8x8 = eb_aom_highbd_dc_top_predictor_8x8_c;

    // eb_aom_highbd_h_predictor
    eb_aom_highbd_h_predictor_16x4 = eb_aom_highbd_h_predictor_16x4_c;
    eb_aom_highbd_h_predictor_16x64 = eb_aom_highbd_h_predictor_16x64_c;
    eb_aom_highbd_h_predictor_16x8 = eb_aom_highbd_h_predictor_16x8_c;
    eb_aom_highbd_h_predictor_32x16 = eb_aom_highbd_h_predictor_32x16_c;
    eb_aom_highbd_h_predictor_32x32 = eb_aom_highbd_h_predictor_32x32_c;
    eb_aom_highbd_h_predictor_32x64 = eb_aom_highbd_h_predictor_32x64_c;
    eb_aom_highbd_h_predictor_32x8 = eb_aom_highbd_h_predictor_32x8_c;
    eb_aom_highbd_h_predictor_4x16 = eb_aom_highbd_h_predictor_4x16_c;
    eb_aom_highbd_h_predictor_4x4 = eb_aom_highbd_h_predictor_4x4_c;
    eb_aom_highbd_h_predictor_4x8 = eb_aom_highbd_h_predictor_4x8_c;
    eb_aom_highbd_h_predictor_64x16 = eb_aom_highbd_h_predictor_64x16_c;
    eb_aom_highbd_h_predictor_64x32 = eb_aom_highbd_h_predictor_64x32_c;
    eb_aom_highbd_h_predictor_8x32 = eb_aom_highbd_h_predictor_8x32_c;
    eb_aom_highbd_h_predictor_64x64 = eb_aom_highbd_h_predictor_64x64_c;
    eb_aom_highbd_h_predictor_8x16 = eb_aom_highbd_h_predictor_8x16_c;
    eb_aom_highbd_h_predictor_8x4 = eb_aom_highbd_h_predictor_8x4_c;
    eb_aom_highbd_h_predictor_8x8 = eb_aom_highbd_h_predictor_8x8_c;
    eb_aom_highbd_h_predictor_16x16 = eb_aom_highbd_h_predictor_16x16_c;
    eb_aom_highbd_h_predictor_16x32 = eb_aom_highbd_h_predictor_16x32_c;
    eb_log2f = log2f_32;
    eb_memcpy = eb_memcpy_c;
#ifdef ARCH_X86
    flags &= get_cpu_flags_to_use();
    if (flags & HAS_SSE4_1) aom_blend_a64_mask = aom_blend_a64_mask_sse4_1;
    if (flags & HAS_AVX2) aom_blend_a64_mask = aom_blend_a64_mask_avx2;
    if (flags & HAS_SSE4_1) aom_blend_a64_hmask = aom_blend_a64_hmask_sse4_1;
    if (flags & HAS_SSE4_1) aom_blend_a64_vmask = aom_blend_a64_vmask_sse4_1;
    if (flags & HAS_SSE4_1) aom_highbd_blend_a64_mask = aom_highbd_blend_a64_mask_sse4_1;
    if (flags & HAS_SSE4_1) aom_highbd_blend_a64_hmask = aom_highbd_blend_a64_hmask_sse4_1;
    if (flags & HAS_SSE4_1) aom_highbd_blend_a64_vmask = aom_highbd_blend_a64_vmask_sse4_1;
    if (flags & HAS_SSE4_1) eb_aom_highbd_blend_a64_vmask = eb_aom_highbd_blend_a64_vmask_sse4_1;
    if (flags & HAS_SSE4_1) eb_aom_highbd_blend_a64_hmask = eb_aom_highbd_blend_a64_hmask_sse4_1;
    if (flags & HAS_AVX2) eb_cfl_predict_lbd = eb_cfl_predict_lbd_avx2;
    if (flags & HAS_AVX2) eb_cfl_predict_hbd = eb_cfl_predict_hbd_avx2;
    if (flags & HAS_SSE4_1) eb_av1_filter_intra_predictor = eb_av1_filter_intra_predictor_sse4_1;
    if (flags & HAS_SSE4_1) eb_av1_filter_intra_edge_high = eb_av1_filter_intra_edge_high_sse4_1;
    if (flags & HAS_SSE4_1) eb_av1_filter_intra_edge = eb_av1_filter_intra_edge_sse4_1;
    if (flags & HAS_SSE4_1) eb_av1_upsample_intra_edge = eb_av1_upsample_intra_edge_sse4_1;
    if (flags & HAS_AVX2) av1_build_compound_diffwtd_mask_d16 = av1_build_compound_diffwtd_mask_d16_avx2;
    if (flags & HAS_AVX2) eb_av1_highbd_wiener_convolve_add_src = eb_av1_highbd_wiener_convolve_add_src_avx2;
    if (flags & HAS_AVX2) eb_apply_selfguided_restoration = eb_apply_selfguided_restoration_avx2;
    if (flags & HAS_AVX2) eb_av1_selfguided_restoration = eb_av1_selfguided_restoration_avx2;
    if (flags & HAS_AVX2) eb_av1_inv_txfm2d_add_4x4 = eb_av1_inv_txfm2d_add_4x4_avx2;
    if (flags & HAS_AVX2) eb_av1_inv_txfm2d_add_8x8 = eb_av1_inv_txfm2d_add_8x8_avx2;
    if (flags & HAS_AVX2) eb_av1_inv_txfm2d_add_8x16 = eb_av1_highbd_inv_txfm_add_avx2;
    if (flags & HAS_AVX2) eb_av1_inv_txfm2d_add_16x8 = eb_av1_highbd_inv_txfm_add_avx2;
    if (flags & HAS_AVX2) eb_av1_inv_txfm2d_add_32x8 = eb_av1_highbd_inv_txfm_add_avx2;
    if (flags & HAS_AVX2) eb_av1_inv_txfm2d_add_8x32 = eb_av1_highbd_inv_txfm_add_avx2;
    if (flags & HAS_SSE4_1) eb_av1_inv_txfm2d_add_4x8 = eb_av1_inv_txfm2d_add_4x8_sse4_1;
    if (flags & HAS_SSE4_1) eb_av1_inv_txfm2d_add_8x4 = eb_av1_inv_txfm2d_add_8x4_sse4_1;
    if (flags & HAS_SSE4_1) eb_av1_inv_txfm2d_add_4x16 = eb_av1_inv_txfm2d_add_4x16_sse4_1;
    if (flags & HAS_SSE4_1) eb_av1_inv_txfm2d_add_16x4 = eb_av1_inv_txfm2d_add_16x4_sse4_1;
    if (flags & HAS_AVX2) eb_av1_inv_txfm2d_add_16x16 = eb_av1_inv_txfm2d_add_16x16_avx2;
    if (flags & HAS_AVX2) eb_av1_inv_txfm2d_add_32x32 = eb_av1_inv_txfm2d_add_32x32_avx2;
    if (flags & HAS_SSE4_1) eb_av1_inv_txfm2d_add_64x64 = eb_av1_inv_txfm2d_add_64x64_sse4_1;
    if (flags & HAS_AVX2) eb_av1_inv_txfm2d_add_16x64 = eb_av1_highbd_inv_txfm_add_avx2;
    if (flags & HAS_AVX2) eb_av1_inv_txfm2d_add_64x16 = eb_av1_highbd_inv_txfm_add_avx2;
    if (flags & HAS_AVX2) eb_av1_inv_txfm2d_add_32x64 = eb_av1_highbd_inv_txfm_add_avx2;
    if (flags & HAS_AVX2) eb_av1_inv_txfm2d_add_64x32 = eb_av1_highbd_inv_txfm_add_avx2;
    if (flags & HAS_AVX2) eb_av1_inv_txfm2d_add_16x32 = eb_av1_highbd_inv_txfm_add_avx2;
    if (flags & HAS_AVX2) eb_av1_inv_txfm2d_add_32x16 = eb_av1_highbd_inv_txfm_add_avx2;
#ifndef NON_AVX512_SUPPORT
    if (flags & HAS_AVX512F) {
        eb_av1_inv_txfm2d_add_16x16 = eb_av1_inv_txfm2d_add_16x16_avx512;
        eb_av1_inv_txfm2d_add_32x32 = eb_av1_inv_txfm2d_add_32x32_avx512;
        eb_av1_inv_txfm2d_add_64x64 = eb_av1_inv_txfm2d_add_64x64_avx512;
        eb_av1_inv_txfm2d_add_16x64 = eb_av1_inv_txfm2d_add_16x64_avx512;
        eb_av1_inv_txfm2d_add_64x16 = eb_av1_inv_txfm2d_add_64x16_avx512;
        eb_av1_inv_txfm2d_add_32x64 = eb_av1_inv_txfm2d_add_32x64_avx512;
        eb_av1_inv_txfm2d_add_64x32 = eb_av1_inv_txfm2d_add_64x32_avx512;
        eb_av1_inv_txfm2d_add_16x32 = eb_av1_inv_txfm2d_add_16x32_avx512;
        eb_av1_inv_txfm2d_add_32x16 = eb_av1_inv_txfm2d_add_32x16_avx512;
    }
#endif

        if (flags & HAS_SSSE3) eb_av1_inv_txfm_add = eb_av1_inv_txfm_add_ssse3;
        if (flags & HAS_AVX2) eb_av1_inv_txfm_add = eb_av1_inv_txfm_add_avx2;
        SET_AVX2(compressed_packmsb, compressed_packmsb_c, compressed_packmsb_avx2_intrin);
        SET_AVX2(c_pack, c_pack_c, c_pack_avx2_intrin);
        SET_SSE2_AVX2(unpack_avg, unpack_avg_c, unpack_avg_sse2_intrin, unpack_avg_avx2_intrin);
        SET_AVX2(unpack_avg_safe_sub, unpack_avg_safe_sub_c, unpack_avg_safe_sub_avx2_intrin);
        SET_AVX2(un_pack8_bit_data, un_pack8_bit_data_c, eb_enc_un_pack8_bit_data_avx2_intrin);
        SET_AVX2(cfl_luma_subsampling_420_lbd, cfl_luma_subsampling_420_lbd_c, cfl_luma_subsampling_420_lbd_avx2);
        SET_AVX2(cfl_luma_subsampling_420_hbd, cfl_luma_subsampling_420_hbd_c, cfl_luma_subsampling_420_hbd_avx2);
        SET_AVX2(convert_8bit_to_16bit, convert_8bit_to_16bit_c, convert_8bit_to_16bit_avx2);
        SET_AVX2(convert_16bit_to_8bit, convert_16bit_to_8bit_c, convert_16bit_to_8bit_avx2);
        SET_SSE2_AVX2(pack2d_16_bit_src_mul4,
            eb_enc_msb_pack2_d,
            eb_enc_msb_pack2d_sse2_intrin,
            eb_enc_msb_pack2d_avx2_intrin_al);
        SET_SSE2(un_pack2d_16_bit_src_mul4, eb_enc_msb_un_pack2_d, eb_enc_msb_un_pack2d_sse2_intrin);
        SET_AVX2(full_distortion_kernel_cbf_zero32_bits,
            full_distortion_kernel_cbf_zero32_bits_c,
            full_distortion_kernel_cbf_zero32_bits_avx2);
        SET_AVX2(full_distortion_kernel32_bits,
            full_distortion_kernel32_bits_c,
            full_distortion_kernel32_bits_avx2);

        SET_AVX2_AVX512(spatial_full_distortion_kernel,
            spatial_full_distortion_kernel_c,
            spatial_full_distortion_kernel_avx2,
            spatial_full_distortion_kernel_avx512);
        SET_AVX2(full_distortion_kernel16_bits,
            full_distortion_kernel16_bits_c,
            full_distortion_kernel16_bits_avx2);
        SET_AVX2_AVX512(residual_kernel8bit,
            residual_kernel8bit_c,
            residual_kernel8bit_avx2,
            residual_kernel8bit_avx512);

        SET_SSE2(residual_kernel16bit, residual_kernel16bit_c, residual_kernel16bit_sse2_intrin);
        SET_SSE2(picture_average_kernel, picture_average_kernel_c, picture_average_kernel_sse2_intrin);
        SET_SSE2(picture_average_kernel1_line,
            picture_average_kernel1_line_c,
            picture_average_kernel1_line_sse2_intrin);

        SET_SSSE3(avc_style_luma_interpolation_filter,
            avc_style_luma_interpolation_filter_helper_c,
            avc_style_luma_interpolation_filter_helper_ssse3);

        SET_AVX2_AVX512(eb_av1_wiener_convolve_add_src,
            eb_av1_wiener_convolve_add_src_c,
            eb_av1_wiener_convolve_add_src_avx2,
            eb_av1_wiener_convolve_add_src_avx512);

        if (flags & HAS_AVX2) eb_av1_convolve_2d_copy_sr = eb_av1_convolve_2d_copy_sr_avx2;
        if (flags & HAS_AVX2) eb_av1_highbd_convolve_2d_copy_sr = eb_av1_highbd_convolve_2d_copy_sr_avx2;
        if (flags & HAS_AVX2) eb_av1_highbd_jnt_convolve_2d_copy = eb_av1_highbd_jnt_convolve_2d_copy_avx2;
        if (flags & HAS_AVX2) eb_av1_highbd_convolve_y_sr = eb_av1_highbd_convolve_y_sr_avx2;
        if (flags & HAS_AVX2) eb_av1_highbd_convolve_2d_sr = eb_av1_highbd_convolve_2d_sr_avx2;
        if (flags & HAS_AVX2) eb_av1_highbd_jnt_convolve_2d = eb_av1_highbd_jnt_convolve_2d_avx2;
        if (flags & HAS_AVX2) eb_av1_highbd_jnt_convolve_x = eb_av1_highbd_jnt_convolve_x_avx2;
        if (flags & HAS_AVX2) eb_av1_highbd_jnt_convolve_y = eb_av1_highbd_jnt_convolve_y_avx2;
        if (flags & HAS_AVX2) eb_av1_highbd_convolve_x_sr = eb_av1_highbd_convolve_x_sr_avx2;
        SET_AVX2_AVX512(eb_av1_convolve_2d_sr,
            eb_av1_convolve_2d_sr_c,
            eb_av1_convolve_2d_sr_avx2,
            eb_av1_convolve_2d_sr_avx512);
        SET_AVX2_AVX512(eb_av1_convolve_2d_copy_sr,
            eb_av1_convolve_2d_copy_sr_c,
            eb_av1_convolve_2d_copy_sr_avx2,
            eb_av1_convolve_2d_copy_sr_avx512);
        SET_AVX2_AVX512(eb_av1_convolve_x_sr,
            eb_av1_convolve_x_sr_c,
            eb_av1_convolve_x_sr_avx2,
            eb_av1_convolve_x_sr_avx512);
        SET_AVX2_AVX512(eb_av1_convolve_y_sr,
            eb_av1_convolve_y_sr_c,
            eb_av1_convolve_y_sr_avx2,
            eb_av1_convolve_y_sr_avx512);
        SET_AVX2_AVX512(eb_av1_jnt_convolve_2d,
            eb_av1_jnt_convolve_2d_c,
            eb_av1_jnt_convolve_2d_avx2,
            eb_av1_jnt_convolve_2d_avx512);
        SET_AVX2_AVX512(eb_av1_jnt_convolve_2d_copy,
            eb_av1_jnt_convolve_2d_copy_c,
            eb_av1_jnt_convolve_2d_copy_avx2,
            eb_av1_jnt_convolve_2d_copy_avx512);
        SET_AVX2_AVX512(eb_av1_jnt_convolve_x,
            eb_av1_jnt_convolve_x_c,
            eb_av1_jnt_convolve_x_avx2,
            eb_av1_jnt_convolve_x_avx512);
        SET_AVX2_AVX512(eb_av1_jnt_convolve_y,
            eb_av1_jnt_convolve_y_c,
            eb_av1_jnt_convolve_y_avx2,
            eb_av1_jnt_convolve_y_avx512);

        if (flags & HAS_AVX2) aom_convolve8_horiz = aom_convolve8_horiz_avx2;
        if (flags & HAS_AVX2) aom_convolve8_vert = aom_convolve8_vert_avx2;
        if (flags & HAS_AVX2) av1_build_compound_diffwtd_mask = av1_build_compound_diffwtd_mask_avx2;
        if (flags & HAS_AVX2) av1_build_compound_diffwtd_mask_highbd = av1_build_compound_diffwtd_mask_highbd_avx2;
        if (flags & HAS_AVX2) av1_wedge_sse_from_residuals = av1_wedge_sse_from_residuals_avx2;
        if (flags & HAS_AVX2) aom_subtract_block = aom_subtract_block_avx2;
        if (flags & HAS_AVX2) aom_lowbd_blend_a64_d16_mask = aom_lowbd_blend_a64_d16_mask_avx2;
        if (flags & HAS_AVX2) aom_highbd_blend_a64_d16_mask = aom_highbd_blend_a64_d16_mask_avx2;
        if (flags & HAS_AVX2) aom_highbd_subtract_block = aom_highbd_subtract_block_sse2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_v_predictor_16x16 = eb_aom_highbd_smooth_v_predictor_16x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_v_predictor_16x32 = eb_aom_highbd_smooth_v_predictor_16x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_v_predictor_16x4 = eb_aom_highbd_smooth_v_predictor_16x4_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_v_predictor_16x64 = eb_aom_highbd_smooth_v_predictor_16x64_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_v_predictor_16x8 = eb_aom_highbd_smooth_v_predictor_16x8_avx2;
        if (flags & HAS_SSSE3) eb_aom_highbd_smooth_v_predictor_4x16 = eb_aom_highbd_smooth_v_predictor_4x16_ssse3;
        if (flags & HAS_SSSE3) eb_aom_highbd_smooth_v_predictor_4x4 = eb_aom_highbd_smooth_v_predictor_4x4_ssse3;
        if (flags & HAS_SSSE3) eb_aom_highbd_smooth_v_predictor_4x8 = eb_aom_highbd_smooth_v_predictor_4x8_ssse3;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_v_predictor_8x16 = eb_aom_highbd_smooth_v_predictor_8x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_v_predictor_8x32 = eb_aom_highbd_smooth_v_predictor_8x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_v_predictor_8x4 = eb_aom_highbd_smooth_v_predictor_8x4_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_v_predictor_8x8 = eb_aom_highbd_smooth_v_predictor_8x8_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_v_predictor_32x8 = eb_aom_highbd_smooth_v_predictor_32x8_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_v_predictor_32x16 = eb_aom_highbd_smooth_v_predictor_32x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_v_predictor_32x32 = eb_aom_highbd_smooth_v_predictor_32x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_v_predictor_32x64 = eb_aom_highbd_smooth_v_predictor_32x64_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_v_predictor_64x16 = eb_aom_highbd_smooth_v_predictor_64x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_v_predictor_64x32 = eb_aom_highbd_smooth_v_predictor_64x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_v_predictor_64x64 = eb_aom_highbd_smooth_v_predictor_64x64_avx2;

#ifndef NON_AVX512_SUPPORT
        if (flags & HAS_AVX512F) {
            eb_aom_highbd_smooth_v_predictor_32x8 = aom_highbd_smooth_v_predictor_32x8_avx512;
            eb_aom_highbd_smooth_v_predictor_32x16 = aom_highbd_smooth_v_predictor_32x16_avx512;
            eb_aom_highbd_smooth_v_predictor_32x32 = aom_highbd_smooth_v_predictor_32x32_avx512;
            eb_aom_highbd_smooth_v_predictor_32x64 = aom_highbd_smooth_v_predictor_32x64_avx512;
            eb_aom_highbd_smooth_v_predictor_64x16 = aom_highbd_smooth_v_predictor_64x16_avx512;
            eb_aom_highbd_smooth_v_predictor_64x32 = aom_highbd_smooth_v_predictor_64x32_avx512;
            eb_aom_highbd_smooth_v_predictor_64x64 = aom_highbd_smooth_v_predictor_64x64_avx512;
        }
#endif // !NON_AVX512_SUPPORT

        if (flags & HAS_AVX2) eb_av1_dr_prediction_z1 = eb_av1_dr_prediction_z1_avx2;
        if (flags & HAS_AVX2) eb_av1_dr_prediction_z2 = eb_av1_dr_prediction_z2_avx2;
        if (flags & HAS_AVX2) eb_av1_dr_prediction_z3 = eb_av1_dr_prediction_z3_avx2;
        if (flags & HAS_AVX2) eb_av1_highbd_dr_prediction_z1 = eb_av1_highbd_dr_prediction_z1_avx2;
        if (flags & HAS_AVX2) eb_av1_highbd_dr_prediction_z2 = eb_av1_highbd_dr_prediction_z2_avx2;
        if (flags & HAS_AVX2) eb_av1_highbd_dr_prediction_z3 = eb_av1_highbd_dr_prediction_z3_avx2;
        if (flags & HAS_SSSE3) eb_aom_paeth_predictor_16x16 = eb_aom_paeth_predictor_16x16_ssse3;
        if (flags & HAS_AVX2) eb_aom_paeth_predictor_16x16 = eb_aom_paeth_predictor_16x16_avx2;
        if (flags & HAS_SSSE3) eb_aom_paeth_predictor_16x32 = eb_aom_paeth_predictor_16x32_ssse3;
        if (flags & HAS_AVX2) eb_aom_paeth_predictor_16x32 = eb_aom_paeth_predictor_16x32_avx2;
        if (flags & HAS_SSSE3) eb_aom_paeth_predictor_16x4 = eb_aom_paeth_predictor_16x4_ssse3;
        if (flags & HAS_SSSE3) eb_aom_paeth_predictor_16x64 = eb_aom_paeth_predictor_16x64_ssse3;
        if (flags & HAS_AVX2) eb_aom_paeth_predictor_16x64 = eb_aom_paeth_predictor_16x64_avx2;
        if (flags & HAS_SSSE3) eb_aom_paeth_predictor_16x8 = eb_aom_paeth_predictor_16x8_ssse3;
        if (flags & HAS_AVX2) eb_aom_paeth_predictor_16x8 = eb_aom_paeth_predictor_16x8_avx2;
        if (flags & HAS_SSSE3) eb_aom_paeth_predictor_32x16 = eb_aom_paeth_predictor_32x16_ssse3;
        if (flags & HAS_AVX2) eb_aom_paeth_predictor_32x16 = eb_aom_paeth_predictor_32x16_avx2;
        if (flags & HAS_SSSE3) eb_aom_paeth_predictor_32x32 = eb_aom_paeth_predictor_32x32_ssse3;
        if (flags & HAS_AVX2) eb_aom_paeth_predictor_32x32 = eb_aom_paeth_predictor_32x32_avx2;
        if (flags & HAS_SSSE3) eb_aom_paeth_predictor_32x64 = eb_aom_paeth_predictor_32x64_ssse3;
        if (flags & HAS_AVX2) eb_aom_paeth_predictor_32x64 = eb_aom_paeth_predictor_32x64_avx2;
        if (flags & HAS_SSSE3) eb_aom_paeth_predictor_32x8 = eb_aom_paeth_predictor_32x8_ssse3;
        if (flags & HAS_SSSE3) eb_aom_paeth_predictor_4x16 = eb_aom_paeth_predictor_4x16_ssse3;
        if (flags & HAS_SSSE3) eb_aom_paeth_predictor_4x4 = eb_aom_paeth_predictor_4x4_ssse3;
        if (flags & HAS_SSSE3) eb_aom_paeth_predictor_4x8 = eb_aom_paeth_predictor_4x8_ssse3;
        if (flags & HAS_SSSE3) eb_aom_paeth_predictor_64x16 = eb_aom_paeth_predictor_64x16_ssse3;
        if (flags & HAS_AVX2) eb_aom_paeth_predictor_64x16 = eb_aom_paeth_predictor_64x16_avx2;
        if (flags & HAS_SSSE3) eb_aom_paeth_predictor_64x32 = eb_aom_paeth_predictor_64x32_ssse3;
        if (flags & HAS_AVX2) eb_aom_paeth_predictor_64x32 = eb_aom_paeth_predictor_64x32_avx2;
        if (flags & HAS_SSSE3) eb_aom_paeth_predictor_64x64 = eb_aom_paeth_predictor_64x64_ssse3;
        if (flags & HAS_AVX2) eb_aom_paeth_predictor_64x64 = eb_aom_paeth_predictor_64x64_avx2;
        if (flags & HAS_SSSE3) eb_aom_paeth_predictor_8x16 = eb_aom_paeth_predictor_8x16_ssse3;
        if (flags & HAS_SSSE3) eb_aom_paeth_predictor_8x32 = eb_aom_paeth_predictor_8x32_ssse3;
        if (flags & HAS_SSSE3) eb_aom_paeth_predictor_8x4 = eb_aom_paeth_predictor_8x4_ssse3;
        if (flags & HAS_SSSE3) eb_aom_paeth_predictor_8x8 = eb_aom_paeth_predictor_8x8_ssse3;
        if (flags & HAS_AVX2) eb_aom_highbd_paeth_predictor_16x16 = eb_aom_highbd_paeth_predictor_16x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_paeth_predictor_16x32 = eb_aom_highbd_paeth_predictor_16x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_paeth_predictor_16x4 = eb_aom_highbd_paeth_predictor_16x4_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_paeth_predictor_16x64 = eb_aom_highbd_paeth_predictor_16x64_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_paeth_predictor_16x8 = eb_aom_highbd_paeth_predictor_16x8_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_paeth_predictor_2x2 = eb_aom_highbd_paeth_predictor_2x2_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_paeth_predictor_32x16 = eb_aom_highbd_paeth_predictor_32x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_paeth_predictor_32x32 = eb_aom_highbd_paeth_predictor_32x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_paeth_predictor_32x64 = eb_aom_highbd_paeth_predictor_32x64_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_paeth_predictor_32x8 = eb_aom_highbd_paeth_predictor_32x8_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_paeth_predictor_4x16 = eb_aom_highbd_paeth_predictor_4x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_paeth_predictor_4x4 = eb_aom_highbd_paeth_predictor_4x4_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_paeth_predictor_4x8 = eb_aom_highbd_paeth_predictor_4x8_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_paeth_predictor_64x16 = eb_aom_highbd_paeth_predictor_64x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_paeth_predictor_64x32 = eb_aom_highbd_paeth_predictor_64x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_paeth_predictor_64x64 = eb_aom_highbd_paeth_predictor_64x64_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_paeth_predictor_8x16 = eb_aom_highbd_paeth_predictor_8x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_paeth_predictor_8x32 = eb_aom_highbd_paeth_predictor_8x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_paeth_predictor_8x4 = eb_aom_highbd_paeth_predictor_8x4_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_paeth_predictor_8x8 = eb_aom_highbd_paeth_predictor_8x8_avx2;
        if (flags & HAS_SSE2) aom_sum_squares_i16 = aom_sum_squares_i16_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_predictor_4x4 = eb_aom_dc_predictor_4x4_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_predictor_8x8 = eb_aom_dc_predictor_8x8_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_predictor_16x16 = eb_aom_dc_predictor_16x16_sse2;
        if (flags & HAS_AVX2) eb_aom_dc_predictor_32x32 = eb_aom_dc_predictor_32x32_avx2;
        if (flags & HAS_AVX2) eb_aom_dc_predictor_64x64 = eb_aom_dc_predictor_64x64_avx2;
        if (flags & HAS_AVX2) eb_aom_dc_predictor_32x16 = eb_aom_dc_predictor_32x16_avx2;
        if (flags & HAS_AVX2) eb_aom_dc_predictor_32x64 = eb_aom_dc_predictor_32x64_avx2;
        if (flags & HAS_AVX2) eb_aom_dc_predictor_64x16 = eb_aom_dc_predictor_64x16_avx2;
        if (flags & HAS_SSE2) eb_aom_dc_predictor_8x16 = eb_aom_dc_predictor_8x16_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_predictor_8x32 = eb_aom_dc_predictor_8x32_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_predictor_8x4 = eb_aom_dc_predictor_8x4_sse2;
        if (flags & HAS_AVX2) eb_aom_dc_predictor_64x32 = eb_aom_dc_predictor_64x32_avx2;
        if (flags & HAS_SSE2) eb_aom_dc_predictor_16x32 = eb_aom_dc_predictor_16x32_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_predictor_16x4 = eb_aom_dc_predictor_16x4_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_predictor_16x64 = eb_aom_dc_predictor_16x64_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_predictor_16x8 = eb_aom_dc_predictor_16x8_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_predictor_32x8 = eb_aom_dc_predictor_32x8_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_predictor_4x16 = eb_aom_dc_predictor_4x16_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_predictor_4x8 = eb_aom_dc_predictor_4x8_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_top_predictor_4x4 = eb_aom_dc_top_predictor_4x4_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_top_predictor_8x8 = eb_aom_dc_top_predictor_8x8_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_top_predictor_16x16 = eb_aom_dc_top_predictor_16x16_sse2;
        if (flags & HAS_AVX2) eb_aom_dc_top_predictor_32x32 = eb_aom_dc_top_predictor_32x32_avx2;
        if (flags & HAS_AVX2) eb_aom_dc_top_predictor_64x64 = eb_aom_dc_top_predictor_64x64_avx2;
        if (flags & HAS_SSE2) eb_aom_dc_top_predictor_16x32 = eb_aom_dc_top_predictor_16x32_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_top_predictor_16x4 = eb_aom_dc_top_predictor_16x4_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_top_predictor_16x64 = eb_aom_dc_top_predictor_16x64_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_top_predictor_16x8 = eb_aom_dc_top_predictor_16x8_sse2;
        if (flags & HAS_AVX2) eb_aom_dc_top_predictor_32x16 = eb_aom_dc_top_predictor_32x16_avx2;
        if (flags & HAS_AVX2) eb_aom_dc_top_predictor_32x64 = eb_aom_dc_top_predictor_32x64_avx2;
        if (flags & HAS_SSE2) eb_aom_dc_top_predictor_32x8 = eb_aom_dc_top_predictor_32x8_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_top_predictor_4x16 = eb_aom_dc_top_predictor_4x16_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_top_predictor_4x8 = eb_aom_dc_top_predictor_4x8_sse2;
        if (flags & HAS_AVX2) eb_aom_dc_top_predictor_64x16 = eb_aom_dc_top_predictor_64x16_avx2;
        if (flags & HAS_AVX2) eb_aom_dc_top_predictor_64x32 = eb_aom_dc_top_predictor_64x32_avx2;
        if (flags & HAS_SSE2) eb_aom_dc_top_predictor_8x16 = eb_aom_dc_top_predictor_8x16_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_top_predictor_8x32 = eb_aom_dc_top_predictor_8x32_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_top_predictor_8x4 = eb_aom_dc_top_predictor_8x4_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_left_predictor_4x4 = eb_aom_dc_left_predictor_4x4_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_left_predictor_8x8 = eb_aom_dc_left_predictor_8x8_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_left_predictor_16x16 = eb_aom_dc_left_predictor_16x16_sse2;
        if (flags & HAS_AVX2) eb_aom_dc_left_predictor_32x32 = eb_aom_dc_left_predictor_32x32_avx2;
        if (flags & HAS_AVX2) eb_aom_dc_left_predictor_64x64 = eb_aom_dc_left_predictor_64x64_avx2;
        if (flags & HAS_SSE2) eb_aom_dc_left_predictor_16x32 = eb_aom_dc_left_predictor_16x32_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_left_predictor_16x4 = eb_aom_dc_left_predictor_16x4_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_left_predictor_16x64 = eb_aom_dc_left_predictor_16x64_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_left_predictor_16x8 = eb_aom_dc_left_predictor_16x8_sse2;
        if (flags & HAS_AVX2) eb_aom_dc_left_predictor_32x16 = eb_aom_dc_left_predictor_32x16_avx2;
        if (flags & HAS_AVX2) eb_aom_dc_left_predictor_32x64 = eb_aom_dc_left_predictor_32x64_avx2;
        if (flags & HAS_AVX2) eb_aom_dc_left_predictor_64x16 = eb_aom_dc_left_predictor_64x16_avx2;
        if (flags & HAS_AVX2) eb_aom_dc_left_predictor_64x32 = eb_aom_dc_left_predictor_64x32_avx2;
        if (flags & HAS_SSE2) eb_aom_dc_left_predictor_32x8 = eb_aom_dc_left_predictor_32x8_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_left_predictor_4x16 = eb_aom_dc_left_predictor_4x16_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_left_predictor_4x8 = eb_aom_dc_left_predictor_4x8_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_left_predictor_8x16 = eb_aom_dc_left_predictor_8x16_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_left_predictor_8x32 = eb_aom_dc_left_predictor_8x32_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_left_predictor_8x4 = eb_aom_dc_left_predictor_8x4_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_128_predictor_4x4 = eb_aom_dc_128_predictor_4x4_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_128_predictor_8x8 = eb_aom_dc_128_predictor_8x8_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_128_predictor_16x16 = eb_aom_dc_128_predictor_16x16_sse2;
        if (flags & HAS_AVX2) eb_aom_dc_128_predictor_32x32 = eb_aom_dc_128_predictor_32x32_avx2;
        if (flags & HAS_AVX2) eb_aom_dc_128_predictor_64x64 = eb_aom_dc_128_predictor_64x64_avx2;
        if (flags & HAS_SSE2) eb_aom_dc_128_predictor_16x32 = eb_aom_dc_128_predictor_16x32_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_128_predictor_16x4 = eb_aom_dc_128_predictor_16x4_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_128_predictor_16x64 = eb_aom_dc_128_predictor_16x64_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_128_predictor_16x8 = eb_aom_dc_128_predictor_16x8_sse2;
        if (flags & HAS_AVX2) eb_aom_dc_128_predictor_32x16 = eb_aom_dc_128_predictor_32x16_avx2;
        if (flags & HAS_AVX2) eb_aom_dc_128_predictor_32x64 = eb_aom_dc_128_predictor_32x64_avx2;
        if (flags & HAS_SSE2) eb_aom_dc_128_predictor_32x8 = eb_aom_dc_128_predictor_32x8_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_128_predictor_4x16 = eb_aom_dc_128_predictor_4x16_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_128_predictor_4x8 = eb_aom_dc_128_predictor_4x8_sse2;
        if (flags & HAS_AVX2) eb_aom_dc_128_predictor_64x16 = eb_aom_dc_128_predictor_64x16_avx2;
        if (flags & HAS_AVX2) eb_aom_dc_128_predictor_64x32 = eb_aom_dc_128_predictor_64x32_avx2;
        if (flags & HAS_SSE2) eb_aom_dc_128_predictor_8x16 = eb_aom_dc_128_predictor_8x16_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_128_predictor_8x32 = eb_aom_dc_128_predictor_8x32_sse2;
        if (flags & HAS_SSE2) eb_aom_dc_128_predictor_8x4 = eb_aom_dc_128_predictor_8x4_sse2;
        if (flags & HAS_SSSE3) eb_aom_smooth_h_predictor_16x32 = eb_aom_smooth_h_predictor_16x32_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_h_predictor_16x4 = eb_aom_smooth_h_predictor_16x4_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_h_predictor_16x64 = eb_aom_smooth_h_predictor_16x64_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_h_predictor_16x8 = eb_aom_smooth_h_predictor_16x8_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_h_predictor_32x16 = eb_aom_smooth_h_predictor_32x16_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_h_predictor_32x64 = eb_aom_smooth_h_predictor_32x64_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_h_predictor_32x8 = eb_aom_smooth_h_predictor_32x8_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_h_predictor_4x16 = eb_aom_smooth_h_predictor_4x16_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_h_predictor_4x8 = eb_aom_smooth_h_predictor_4x8_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_h_predictor_64x16 = eb_aom_smooth_h_predictor_64x16_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_h_predictor_64x32 = eb_aom_smooth_h_predictor_64x32_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_h_predictor_8x16 = eb_aom_smooth_h_predictor_8x16_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_h_predictor_8x32 = eb_aom_smooth_h_predictor_8x32_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_h_predictor_8x4 = eb_aom_smooth_h_predictor_8x4_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_h_predictor_64x64 = eb_aom_smooth_h_predictor_64x64_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_h_predictor_32x32 = eb_aom_smooth_h_predictor_32x32_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_h_predictor_16x16 = eb_aom_smooth_h_predictor_16x16_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_h_predictor_8x8 = eb_aom_smooth_h_predictor_8x8_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_h_predictor_4x4 = eb_aom_smooth_h_predictor_4x4_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_v_predictor_16x32 = eb_aom_smooth_v_predictor_16x32_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_v_predictor_16x4 = eb_aom_smooth_v_predictor_16x4_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_v_predictor_16x64 = eb_aom_smooth_v_predictor_16x64_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_v_predictor_16x8 = eb_aom_smooth_v_predictor_16x8_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_v_predictor_32x16 = eb_aom_smooth_v_predictor_32x16_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_v_predictor_32x64 = eb_aom_smooth_v_predictor_32x64_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_v_predictor_32x8 = eb_aom_smooth_v_predictor_32x8_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_v_predictor_4x16 = eb_aom_smooth_v_predictor_4x16_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_v_predictor_4x8 = eb_aom_smooth_v_predictor_4x8_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_v_predictor_64x16 = eb_aom_smooth_v_predictor_64x16_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_v_predictor_64x32 = eb_aom_smooth_v_predictor_64x32_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_v_predictor_8x16 = eb_aom_smooth_v_predictor_8x16_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_v_predictor_8x32 = eb_aom_smooth_v_predictor_8x32_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_v_predictor_8x4 = eb_aom_smooth_v_predictor_8x4_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_v_predictor_64x64 = eb_aom_smooth_v_predictor_64x64_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_v_predictor_32x32 = eb_aom_smooth_v_predictor_32x32_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_v_predictor_16x16 = eb_aom_smooth_v_predictor_16x16_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_v_predictor_8x8 = eb_aom_smooth_v_predictor_8x8_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_v_predictor_4x4 = eb_aom_smooth_v_predictor_4x4_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_predictor_16x32 = eb_aom_smooth_predictor_16x32_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_predictor_16x4 = eb_aom_smooth_predictor_16x4_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_predictor_16x64 = eb_aom_smooth_predictor_16x64_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_predictor_16x8 = eb_aom_smooth_predictor_16x8_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_predictor_32x16 = eb_aom_smooth_predictor_32x16_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_predictor_32x64 = eb_aom_smooth_predictor_32x64_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_predictor_32x8 = eb_aom_smooth_predictor_32x8_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_predictor_4x16 = eb_aom_smooth_predictor_4x16_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_predictor_4x8 = eb_aom_smooth_predictor_4x8_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_predictor_64x16 = eb_aom_smooth_predictor_64x16_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_predictor_64x32 = eb_aom_smooth_predictor_64x32_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_predictor_8x16 = eb_aom_smooth_predictor_8x16_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_predictor_8x32 = eb_aom_smooth_predictor_8x32_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_predictor_8x4 = eb_aom_smooth_predictor_8x4_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_predictor_64x64 = eb_aom_smooth_predictor_64x64_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_predictor_32x32 = eb_aom_smooth_predictor_32x32_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_predictor_16x16 = eb_aom_smooth_predictor_16x16_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_predictor_8x8 = eb_aom_smooth_predictor_8x8_ssse3;
        if (flags & HAS_SSSE3) eb_aom_smooth_predictor_4x4 = eb_aom_smooth_predictor_4x4_ssse3;
        if (flags & HAS_SSE2) eb_aom_v_predictor_4x4 = eb_aom_v_predictor_4x4_sse2;
        if (flags & HAS_SSE2) eb_aom_v_predictor_8x8 = eb_aom_v_predictor_8x8_sse2;
        if (flags & HAS_SSE2) eb_aom_v_predictor_16x16 = eb_aom_v_predictor_16x16_sse2;
        if (flags & HAS_AVX2) eb_aom_v_predictor_32x32 = eb_aom_v_predictor_32x32_avx2;
        if (flags & HAS_AVX2) eb_aom_v_predictor_64x64 = eb_aom_v_predictor_64x64_avx2;
        if (flags & HAS_SSE2) eb_aom_v_predictor_16x32 = eb_aom_v_predictor_16x32_sse2;
        if (flags & HAS_SSE2) eb_aom_v_predictor_16x4 = eb_aom_v_predictor_16x4_sse2;
        if (flags & HAS_SSE2) eb_aom_v_predictor_16x64 = eb_aom_v_predictor_16x64_sse2;
        if (flags & HAS_SSE2) eb_aom_v_predictor_16x8 = eb_aom_v_predictor_16x8_sse2;
        if (flags & HAS_AVX2) eb_aom_v_predictor_32x16 = eb_aom_v_predictor_32x16_avx2;
        if (flags & HAS_AVX2) eb_aom_v_predictor_32x64 = eb_aom_v_predictor_32x64_avx2;
        if (flags & HAS_SSE2) eb_aom_v_predictor_32x8 = eb_aom_v_predictor_32x8_sse2;
        if (flags & HAS_SSE2) eb_aom_v_predictor_4x16 = eb_aom_v_predictor_4x16_sse2;
        if (flags & HAS_SSE2) eb_aom_v_predictor_4x8 = eb_aom_v_predictor_4x8_sse2;
        if (flags & HAS_AVX2) eb_aom_v_predictor_64x16 = eb_aom_v_predictor_64x16_avx2;
        if (flags & HAS_AVX2) eb_aom_v_predictor_64x32 = eb_aom_v_predictor_64x32_avx2;
        if (flags & HAS_SSE2) eb_aom_v_predictor_8x16 = eb_aom_v_predictor_8x16_sse2;
        if (flags & HAS_SSE2) eb_aom_v_predictor_8x32 = eb_aom_v_predictor_8x32_sse2;
        if (flags & HAS_SSE2) eb_aom_v_predictor_8x4 = eb_aom_v_predictor_8x4_sse2;
        if (flags & HAS_SSE2) eb_aom_h_predictor_4x4 = eb_aom_h_predictor_4x4_sse2;
        if (flags & HAS_SSE2) eb_aom_h_predictor_8x8 = eb_aom_h_predictor_8x8_sse2;
        if (flags & HAS_SSE2) eb_aom_h_predictor_16x16 = eb_aom_h_predictor_16x16_sse2;
        if (flags & HAS_AVX2) eb_aom_h_predictor_32x32 = eb_aom_h_predictor_32x32_avx2;
        if (flags & HAS_SSE2) eb_aom_h_predictor_64x64 = eb_aom_h_predictor_64x64_sse2;
        if (flags & HAS_SSE2) eb_aom_h_predictor_16x32 = eb_aom_h_predictor_16x32_sse2;
        if (flags & HAS_SSE2) eb_aom_h_predictor_16x4 = eb_aom_h_predictor_16x4_sse2;
        if (flags & HAS_SSE2) eb_aom_h_predictor_16x64 = eb_aom_h_predictor_16x64_sse2;
        if (flags & HAS_SSE2) eb_aom_h_predictor_16x8 = eb_aom_h_predictor_16x8_sse2;
        if (flags & HAS_SSE2) eb_aom_h_predictor_32x16 = eb_aom_h_predictor_32x16_sse2;
        if (flags & HAS_SSE2) eb_aom_h_predictor_32x64 = eb_aom_h_predictor_32x64_sse2;
        if (flags & HAS_SSE2) eb_aom_h_predictor_32x8 = eb_aom_h_predictor_32x8_sse2;
        if (flags & HAS_SSE2) eb_aom_h_predictor_4x16 = eb_aom_h_predictor_4x16_sse2;
        if (flags & HAS_SSE2) eb_aom_h_predictor_4x8 = eb_aom_h_predictor_4x8_sse2;
        if (flags & HAS_SSE2) eb_aom_h_predictor_64x16 = eb_aom_h_predictor_64x16_sse2;
        if (flags & HAS_SSE2) eb_aom_h_predictor_64x32 = eb_aom_h_predictor_64x32_sse2;
        if (flags & HAS_SSE2) eb_aom_h_predictor_8x16 = eb_aom_h_predictor_8x16_sse2;
        if (flags & HAS_SSE2) eb_aom_h_predictor_8x32 = eb_aom_h_predictor_8x32_sse2;
        if (flags & HAS_SSE2) eb_aom_h_predictor_8x4 = eb_aom_h_predictor_8x4_sse2;
        if (flags & HAS_AVX2) eb_cdef_find_dir = eb_cdef_find_dir_avx2;
        if (flags & HAS_AVX2) eb_cdef_filter_block = eb_cdef_filter_block_avx2;
        if (flags & HAS_AVX2) eb_copy_rect8_8bit_to_16bit = eb_copy_rect8_8bit_to_16bit_avx2;
        if (flags & HAS_AVX2) eb_cdef_filter_block_8x8_16 = eb_cdef_filter_block_8x8_16_avx2;
#ifndef NON_AVX512_SUPPORT
        if (flags & HAS_AVX512F) {
            eb_cdef_filter_block_8x8_16 = eb_cdef_filter_block_8x8_16_avx512;
        }
#endif
        SET_SSE41(
            eb_av1_highbd_warp_affine, eb_av1_highbd_warp_affine_c, eb_av1_highbd_warp_affine_sse4_1);
        if (flags & HAS_AVX2) eb_av1_warp_affine = eb_av1_warp_affine_avx2;
        SET_SSE2(aom_highbd_lpf_horizontal_14, aom_highbd_lpf_horizontal_14_c, aom_highbd_lpf_horizontal_14_sse2);
        SET_SSE2(aom_highbd_lpf_horizontal_4, aom_highbd_lpf_horizontal_4_c, aom_highbd_lpf_horizontal_4_sse2);
        SET_SSE2(aom_highbd_lpf_horizontal_6, aom_highbd_lpf_horizontal_6_c, aom_highbd_lpf_horizontal_6_sse2);
        SET_SSE2(aom_highbd_lpf_horizontal_8, aom_highbd_lpf_horizontal_8_c, aom_highbd_lpf_horizontal_8_sse2);
        SET_SSE2(aom_highbd_lpf_vertical_14, aom_highbd_lpf_vertical_14_c, aom_highbd_lpf_vertical_14_sse2);
        SET_SSE2(aom_highbd_lpf_vertical_4, aom_highbd_lpf_vertical_4_c, aom_highbd_lpf_vertical_4_sse2);
        SET_SSE2(aom_highbd_lpf_vertical_6, aom_highbd_lpf_vertical_6_c, aom_highbd_lpf_vertical_6_sse2);
        SET_SSE2(aom_highbd_lpf_vertical_8, aom_highbd_lpf_vertical_8_c, aom_highbd_lpf_vertical_8_sse2);
        SET_SSE2(aom_lpf_horizontal_14, aom_lpf_horizontal_14_c, aom_lpf_horizontal_14_sse2);
        SET_SSE2(aom_lpf_horizontal_4, aom_lpf_horizontal_4_c, aom_lpf_horizontal_4_sse2);
        SET_SSE2(aom_lpf_horizontal_6, aom_lpf_horizontal_6_c, aom_lpf_horizontal_6_sse2);
        SET_SSE2(aom_lpf_horizontal_8, aom_lpf_horizontal_8_c, aom_lpf_horizontal_8_sse2);
        SET_SSE2(aom_lpf_vertical_14, aom_lpf_vertical_14_c, aom_lpf_vertical_14_sse2);
        SET_SSE2(aom_lpf_vertical_4, aom_lpf_vertical_4_c, aom_lpf_vertical_4_sse2);
        SET_SSE2(aom_lpf_vertical_6, aom_lpf_vertical_6_c, aom_lpf_vertical_6_sse2);
        SET_SSE2(aom_lpf_vertical_8, aom_lpf_vertical_8_c, aom_lpf_vertical_8_sse2);
        if (flags & HAS_AVX2) eb_aom_highbd_v_predictor_16x16 = eb_aom_highbd_v_predictor_16x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_v_predictor_16x32 = eb_aom_highbd_v_predictor_16x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_v_predictor_16x4 = eb_aom_highbd_v_predictor_16x4_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_v_predictor_16x64 = eb_aom_highbd_v_predictor_16x64_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_v_predictor_16x8 = eb_aom_highbd_v_predictor_16x8_avx2;
        if (flags & HAS_SSE2) eb_aom_highbd_v_predictor_4x16 = eb_aom_highbd_v_predictor_4x16_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_v_predictor_4x4 = eb_aom_highbd_v_predictor_4x4_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_v_predictor_4x8 = eb_aom_highbd_v_predictor_4x8_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_v_predictor_8x32 = eb_aom_highbd_v_predictor_8x32_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_v_predictor_8x16 = eb_aom_highbd_v_predictor_8x16_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_v_predictor_8x4 = eb_aom_highbd_v_predictor_8x4_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_v_predictor_8x8 = eb_aom_highbd_v_predictor_8x8_sse2;
        if (flags & HAS_AVX2) eb_aom_highbd_v_predictor_32x8 = eb_aom_highbd_v_predictor_32x8_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_v_predictor_32x16 = eb_aom_highbd_v_predictor_32x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_v_predictor_32x32 = eb_aom_highbd_v_predictor_32x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_v_predictor_32x64 = eb_aom_highbd_v_predictor_32x64_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_v_predictor_64x16 = eb_aom_highbd_v_predictor_64x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_v_predictor_64x32 = eb_aom_highbd_v_predictor_64x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_v_predictor_64x64 = eb_aom_highbd_v_predictor_64x64_avx2;
#ifndef NON_AVX512_SUPPORT
        if (flags & HAS_AVX512F) {
            eb_aom_highbd_v_predictor_32x8 = aom_highbd_v_predictor_32x8_avx512;
            eb_aom_highbd_v_predictor_32x16 = aom_highbd_v_predictor_32x16_avx512;
            eb_aom_highbd_v_predictor_32x32 = aom_highbd_v_predictor_32x32_avx512;
            eb_aom_highbd_v_predictor_32x64 = aom_highbd_v_predictor_32x64_avx512;
            eb_aom_highbd_v_predictor_64x16 = aom_highbd_v_predictor_64x16_avx512;
            eb_aom_highbd_v_predictor_64x32 = aom_highbd_v_predictor_64x32_avx512;
            eb_aom_highbd_v_predictor_64x64 = aom_highbd_v_predictor_64x64_avx512;
        }
#endif // !NON_AVX512_SUPPORT

        if (flags & HAS_AVX2) eb_aom_highbd_smooth_predictor_16x16 = eb_aom_highbd_smooth_predictor_16x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_predictor_16x32 = eb_aom_highbd_smooth_predictor_16x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_predictor_16x4 = eb_aom_highbd_smooth_predictor_16x4_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_predictor_16x64 = eb_aom_highbd_smooth_predictor_16x64_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_predictor_16x8 = eb_aom_highbd_smooth_predictor_16x8_avx2;
        if (flags & HAS_SSSE3) eb_aom_highbd_smooth_predictor_4x16 = eb_aom_highbd_smooth_predictor_4x16_ssse3;
        if (flags & HAS_SSSE3) eb_aom_highbd_smooth_predictor_4x4 = eb_aom_highbd_smooth_predictor_4x4_ssse3;
        if (flags & HAS_SSSE3) eb_aom_highbd_smooth_predictor_4x8 = eb_aom_highbd_smooth_predictor_4x8_ssse3;

        if (flags & HAS_AVX2) eb_aom_highbd_smooth_predictor_8x16 = eb_aom_highbd_smooth_predictor_8x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_predictor_8x32 = eb_aom_highbd_smooth_predictor_8x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_predictor_8x4 = eb_aom_highbd_smooth_predictor_8x4_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_predictor_8x8 = eb_aom_highbd_smooth_predictor_8x8_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_predictor_32x8 = eb_aom_highbd_smooth_predictor_32x8_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_predictor_32x16 = eb_aom_highbd_smooth_predictor_32x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_predictor_32x32 = eb_aom_highbd_smooth_predictor_32x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_predictor_32x64 = eb_aom_highbd_smooth_predictor_32x64_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_predictor_64x16 = eb_aom_highbd_smooth_predictor_64x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_predictor_64x32 = eb_aom_highbd_smooth_predictor_64x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_predictor_64x64 = eb_aom_highbd_smooth_predictor_64x64_avx2;

#ifndef NON_AVX512_SUPPORT
        if (flags & HAS_AVX512F) {
            eb_aom_highbd_smooth_predictor_32x8 = aom_highbd_smooth_predictor_32x8_avx512;
            eb_aom_highbd_smooth_predictor_32x16 = aom_highbd_smooth_predictor_32x16_avx512;
            eb_aom_highbd_smooth_predictor_32x32 = aom_highbd_smooth_predictor_32x32_avx512;
            eb_aom_highbd_smooth_predictor_32x64 = aom_highbd_smooth_predictor_32x64_avx512;
            eb_aom_highbd_smooth_predictor_64x16 = aom_highbd_smooth_predictor_64x16_avx512;
            eb_aom_highbd_smooth_predictor_64x32 = aom_highbd_smooth_predictor_64x32_avx512;
            eb_aom_highbd_smooth_predictor_64x64 = aom_highbd_smooth_predictor_64x64_avx512;
        }
#endif // !NON_AVX512_SUPPORT

        if (flags & HAS_AVX2) eb_aom_highbd_smooth_h_predictor_16x16 = eb_aom_highbd_smooth_h_predictor_16x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_h_predictor_16x32 = eb_aom_highbd_smooth_h_predictor_16x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_h_predictor_16x4 = eb_aom_highbd_smooth_h_predictor_16x4_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_h_predictor_16x64 = eb_aom_highbd_smooth_h_predictor_16x64_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_h_predictor_16x8 = eb_aom_highbd_smooth_h_predictor_16x8_avx2;
        if (flags & HAS_SSSE3) eb_aom_highbd_smooth_h_predictor_4x16 = eb_aom_highbd_smooth_h_predictor_4x16_ssse3;
        if (flags & HAS_SSSE3) eb_aom_highbd_smooth_h_predictor_4x4 = eb_aom_highbd_smooth_h_predictor_4x4_ssse3;
        if (flags & HAS_SSSE3) eb_aom_highbd_smooth_h_predictor_4x8 = eb_aom_highbd_smooth_h_predictor_4x8_ssse3;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_h_predictor_8x16 = eb_aom_highbd_smooth_h_predictor_8x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_h_predictor_8x32 = eb_aom_highbd_smooth_h_predictor_8x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_h_predictor_8x4 = eb_aom_highbd_smooth_h_predictor_8x4_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_h_predictor_8x8 = eb_aom_highbd_smooth_h_predictor_8x8_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_h_predictor_32x8 = eb_aom_highbd_smooth_h_predictor_32x8_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_h_predictor_32x16 = eb_aom_highbd_smooth_h_predictor_32x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_h_predictor_32x32 = eb_aom_highbd_smooth_h_predictor_32x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_h_predictor_32x64 = eb_aom_highbd_smooth_h_predictor_32x64_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_h_predictor_64x16 = eb_aom_highbd_smooth_h_predictor_64x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_h_predictor_64x32 = eb_aom_highbd_smooth_h_predictor_64x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_smooth_h_predictor_64x64 = eb_aom_highbd_smooth_h_predictor_64x64_avx2;

#ifndef NON_AVX512_SUPPORT
        if (flags & HAS_AVX512F) {
            eb_aom_highbd_smooth_h_predictor_32x8 = aom_highbd_smooth_h_predictor_32x8_avx512;
            eb_aom_highbd_smooth_h_predictor_32x16 = aom_highbd_smooth_h_predictor_32x16_avx512;
            eb_aom_highbd_smooth_h_predictor_32x32 = aom_highbd_smooth_h_predictor_32x32_avx512;
            eb_aom_highbd_smooth_h_predictor_32x64 = aom_highbd_smooth_h_predictor_32x64_avx512;
            eb_aom_highbd_smooth_h_predictor_64x16 = aom_highbd_smooth_h_predictor_64x16_avx512;
            eb_aom_highbd_smooth_h_predictor_64x32 = aom_highbd_smooth_h_predictor_64x32_avx512;
            eb_aom_highbd_smooth_h_predictor_64x64 = aom_highbd_smooth_h_predictor_64x64_avx512;
        }
#endif

        //aom_highbd_dc_128_predictor
        if (flags & HAS_AVX2) eb_aom_highbd_dc_128_predictor_16x16 = eb_aom_highbd_dc_128_predictor_16x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_128_predictor_16x32 = eb_aom_highbd_dc_128_predictor_16x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_128_predictor_16x4 = eb_aom_highbd_dc_128_predictor_16x4_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_128_predictor_16x64 = eb_aom_highbd_dc_128_predictor_16x64_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_128_predictor_16x8 = eb_aom_highbd_dc_128_predictor_16x8_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_128_predictor_32x16 = eb_aom_highbd_dc_128_predictor_32x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_128_predictor_32x32 = eb_aom_highbd_dc_128_predictor_32x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_128_predictor_32x64 = eb_aom_highbd_dc_128_predictor_32x64_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_128_predictor_32x8 = eb_aom_highbd_dc_128_predictor_32x8_avx2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_128_predictor_4x16 = eb_aom_highbd_dc_128_predictor_4x16_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_128_predictor_4x4 = eb_aom_highbd_dc_128_predictor_4x4_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_128_predictor_4x8 = eb_aom_highbd_dc_128_predictor_4x8_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_128_predictor_8x32 = eb_aom_highbd_dc_128_predictor_8x32_sse2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_128_predictor_64x16 = eb_aom_highbd_dc_128_predictor_64x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_128_predictor_64x32 = eb_aom_highbd_dc_128_predictor_64x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_128_predictor_64x64 = eb_aom_highbd_dc_128_predictor_64x64_avx2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_128_predictor_8x16 = eb_aom_highbd_dc_128_predictor_8x16_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_128_predictor_8x4 = eb_aom_highbd_dc_128_predictor_8x4_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_128_predictor_8x8 = eb_aom_highbd_dc_128_predictor_8x8_sse2;

        //aom_highbd_dc_left_predictor
        if (flags & HAS_AVX2) eb_aom_highbd_dc_left_predictor_16x16 = eb_aom_highbd_dc_left_predictor_16x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_left_predictor_16x32 = eb_aom_highbd_dc_left_predictor_16x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_left_predictor_16x4 = eb_aom_highbd_dc_left_predictor_16x4_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_left_predictor_16x64 = eb_aom_highbd_dc_left_predictor_16x64_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_left_predictor_16x8 = eb_aom_highbd_dc_left_predictor_16x8_avx2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_left_predictor_4x16 = eb_aom_highbd_dc_left_predictor_4x16_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_left_predictor_4x4 = eb_aom_highbd_dc_left_predictor_4x4_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_left_predictor_4x8 = eb_aom_highbd_dc_left_predictor_4x8_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_left_predictor_8x32 = eb_aom_highbd_dc_left_predictor_8x32_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_left_predictor_8x16 = eb_aom_highbd_dc_left_predictor_8x16_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_left_predictor_8x4 = eb_aom_highbd_dc_left_predictor_8x4_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_left_predictor_8x8 = eb_aom_highbd_dc_left_predictor_8x8_sse2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_left_predictor_32x8 = eb_aom_highbd_dc_left_predictor_32x8_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_left_predictor_32x16 = eb_aom_highbd_dc_left_predictor_32x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_left_predictor_32x32 = eb_aom_highbd_dc_left_predictor_32x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_left_predictor_32x64 = eb_aom_highbd_dc_left_predictor_32x64_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_left_predictor_64x16 = eb_aom_highbd_dc_left_predictor_64x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_left_predictor_64x32 = eb_aom_highbd_dc_left_predictor_64x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_left_predictor_64x64 = eb_aom_highbd_dc_left_predictor_64x64_avx2;

#ifndef NON_AVX512_SUPPORT
        if (flags & HAS_AVX512F) {
            eb_aom_highbd_dc_left_predictor_32x8 = aom_highbd_dc_left_predictor_32x8_avx512;
            eb_aom_highbd_dc_left_predictor_32x16 = aom_highbd_dc_left_predictor_32x16_avx512;
            eb_aom_highbd_dc_left_predictor_32x32 = aom_highbd_dc_left_predictor_32x32_avx512;
            eb_aom_highbd_dc_left_predictor_32x64 = aom_highbd_dc_left_predictor_32x64_avx512;
            eb_aom_highbd_dc_left_predictor_64x16 = aom_highbd_dc_left_predictor_64x16_avx512;
            eb_aom_highbd_dc_left_predictor_64x32 = aom_highbd_dc_left_predictor_64x32_avx512;
            eb_aom_highbd_dc_left_predictor_64x64 = aom_highbd_dc_left_predictor_64x64_avx512;
        }
#endif // !NON_AVX512_SUPPORT
        if (flags & HAS_AVX2) eb_aom_highbd_dc_predictor_16x16 = eb_aom_highbd_dc_predictor_16x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_predictor_16x32 = eb_aom_highbd_dc_predictor_16x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_predictor_16x4 = eb_aom_highbd_dc_predictor_16x4_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_predictor_16x64 = eb_aom_highbd_dc_predictor_16x64_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_predictor_16x8 = eb_aom_highbd_dc_predictor_16x8_avx2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_predictor_4x16 = eb_aom_highbd_dc_predictor_4x16_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_predictor_4x4 = eb_aom_highbd_dc_predictor_4x4_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_predictor_4x8 = eb_aom_highbd_dc_predictor_4x8_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_predictor_8x16 = eb_aom_highbd_dc_predictor_8x16_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_predictor_8x4 = eb_aom_highbd_dc_predictor_8x4_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_predictor_8x8 = eb_aom_highbd_dc_predictor_8x8_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_predictor_8x32 = eb_aom_highbd_dc_predictor_8x32_sse2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_predictor_32x8 = eb_aom_highbd_dc_predictor_32x8_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_predictor_32x16 = eb_aom_highbd_dc_predictor_32x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_predictor_32x32 = eb_aom_highbd_dc_predictor_32x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_predictor_32x64 = eb_aom_highbd_dc_predictor_32x64_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_predictor_64x16 = eb_aom_highbd_dc_predictor_64x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_predictor_64x32 = eb_aom_highbd_dc_predictor_64x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_predictor_64x64 = eb_aom_highbd_dc_predictor_64x64_avx2;
#ifndef NON_AVX512_SUPPORT
        if (flags & HAS_AVX512F) {
            eb_aom_highbd_dc_predictor_32x8 = aom_highbd_dc_predictor_32x8_avx512;
            eb_aom_highbd_dc_predictor_32x16 = aom_highbd_dc_predictor_32x16_avx512;
            eb_aom_highbd_dc_predictor_32x32 = aom_highbd_dc_predictor_32x32_avx512;
            eb_aom_highbd_dc_predictor_32x64 = aom_highbd_dc_predictor_32x64_avx512;
            eb_aom_highbd_dc_predictor_64x16 = aom_highbd_dc_predictor_64x16_avx512;
            eb_aom_highbd_dc_predictor_64x32 = aom_highbd_dc_predictor_64x32_avx512;
            eb_aom_highbd_dc_predictor_64x64 = aom_highbd_dc_predictor_64x64_avx512;
        }
#endif // !NON_AVX512_SUPPORT
        if (flags & HAS_AVX2) eb_aom_highbd_dc_top_predictor_16x16 = eb_aom_highbd_dc_top_predictor_16x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_top_predictor_16x32 = eb_aom_highbd_dc_top_predictor_16x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_top_predictor_16x4 = eb_aom_highbd_dc_top_predictor_16x4_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_top_predictor_16x64 = eb_aom_highbd_dc_top_predictor_16x64_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_top_predictor_16x8 = eb_aom_highbd_dc_top_predictor_16x8_avx2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_top_predictor_4x16 = eb_aom_highbd_dc_top_predictor_4x16_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_top_predictor_4x4 = eb_aom_highbd_dc_top_predictor_4x4_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_top_predictor_4x8 = eb_aom_highbd_dc_top_predictor_4x8_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_top_predictor_8x16 = eb_aom_highbd_dc_top_predictor_8x16_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_top_predictor_8x4 = eb_aom_highbd_dc_top_predictor_8x4_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_dc_top_predictor_8x8 = eb_aom_highbd_dc_top_predictor_8x8_sse2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_top_predictor_32x8 = eb_aom_highbd_dc_top_predictor_32x8_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_top_predictor_32x16 = eb_aom_highbd_dc_top_predictor_32x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_top_predictor_32x32 = eb_aom_highbd_dc_top_predictor_32x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_top_predictor_32x64 = eb_aom_highbd_dc_top_predictor_32x64_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_top_predictor_64x16 = eb_aom_highbd_dc_top_predictor_64x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_top_predictor_64x32 = eb_aom_highbd_dc_top_predictor_64x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_dc_top_predictor_64x64 = eb_aom_highbd_dc_top_predictor_64x64_avx2;

#ifndef NON_AVX512_SUPPORT
        if (flags & HAS_AVX512F) {
            eb_aom_highbd_dc_top_predictor_32x8 = aom_highbd_dc_top_predictor_32x8_avx512;
            eb_aom_highbd_dc_top_predictor_32x16 = aom_highbd_dc_top_predictor_32x16_avx512;
            eb_aom_highbd_dc_top_predictor_32x32 = aom_highbd_dc_top_predictor_32x32_avx512;
            eb_aom_highbd_dc_top_predictor_32x64 = aom_highbd_dc_top_predictor_32x64_avx512;
            eb_aom_highbd_dc_top_predictor_64x16 = aom_highbd_dc_top_predictor_64x16_avx512;
            eb_aom_highbd_dc_top_predictor_64x32 = aom_highbd_dc_top_predictor_64x32_avx512;
            eb_aom_highbd_dc_top_predictor_64x64 = aom_highbd_dc_top_predictor_64x64_avx512;
        }
#endif
        if (flags & HAS_AVX2) eb_aom_highbd_h_predictor_16x4 = eb_aom_highbd_h_predictor_16x4_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_h_predictor_16x64 = eb_aom_highbd_h_predictor_16x64_avx2;
        if (flags & HAS_SSE2) eb_aom_highbd_h_predictor_16x8 = eb_aom_highbd_h_predictor_16x8_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_h_predictor_4x16 = eb_aom_highbd_h_predictor_4x16_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_h_predictor_4x4 = eb_aom_highbd_h_predictor_4x4_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_h_predictor_4x8 = eb_aom_highbd_h_predictor_4x8_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_h_predictor_8x32 = eb_aom_highbd_h_predictor_8x32_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_h_predictor_8x16 = eb_aom_highbd_h_predictor_8x16_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_h_predictor_8x4 = eb_aom_highbd_h_predictor_8x4_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_h_predictor_8x8 = eb_aom_highbd_h_predictor_8x8_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_h_predictor_16x16 = eb_aom_highbd_h_predictor_16x16_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_h_predictor_16x32 = eb_aom_highbd_h_predictor_16x32_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_h_predictor_32x16 = eb_aom_highbd_h_predictor_32x16_sse2;
        if (flags & HAS_SSE2) eb_aom_highbd_h_predictor_32x32 = eb_aom_highbd_h_predictor_32x32_sse2;
        if (flags & HAS_AVX2) eb_aom_highbd_h_predictor_32x64 = eb_aom_highbd_h_predictor_32x64_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_h_predictor_32x8 = eb_aom_highbd_h_predictor_32x8_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_h_predictor_64x16 = eb_aom_highbd_h_predictor_64x16_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_h_predictor_64x32 = eb_aom_highbd_h_predictor_64x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_h_predictor_64x64 = eb_aom_highbd_h_predictor_64x64_avx2;
        if (flags & HAS_SSE2) eb_log2f = Log2f_ASM;
        if (flags & HAS_SSE2) eb_memcpy = eb_memcpy_intrin_sse;
#ifndef NON_AVX512_SUPPORT
        if (flags & HAS_AVX512F) {
            eb_aom_highbd_h_predictor_32x16 = aom_highbd_h_predictor_32x16_avx512;
            eb_aom_highbd_h_predictor_32x32 = aom_highbd_h_predictor_32x32_avx512;
            eb_aom_highbd_h_predictor_32x64 = aom_highbd_h_predictor_32x64_avx512;
            eb_aom_highbd_h_predictor_32x8 = aom_highbd_h_predictor_32x8_avx512;
            eb_aom_highbd_h_predictor_64x16 = aom_highbd_h_predictor_64x16_avx512;
            eb_aom_highbd_h_predictor_64x32 = aom_highbd_h_predictor_64x32_avx512;
            eb_aom_highbd_h_predictor_64x64 = aom_highbd_h_predictor_64x64_avx512;
        }
#endif

#endif


}
