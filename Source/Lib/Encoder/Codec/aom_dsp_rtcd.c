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

#include "aom_dsp_rtcd.h"
#include "EbComputeSAD_C.h"
#include "EbPictureAnalysisProcess.h"
#include "EbTemporalFiltering.h"
#include "EbComputeSAD.h"
#include "EbMotionEstimation.h"
#include "EbPictureOperators.h"
#include "EbComputeMean.h"
#include "EbMeSadCalculation.h"

/**********************************************
 * global function pointer variable definition 
 **********************************************/
int64_t(*aom_sse)(const uint8_t *a, int a_stride, const uint8_t *b, int b_stride, int width, int height);
int64_t(*aom_highbd_sse)(const uint8_t *a8, int a_stride, const uint8_t *b8, int b_stride, int width, int height);
void(*av1_wedge_compute_delta_squares)(int16_t *d, const int16_t *a, const int16_t *b, int N);
int8_t(*av1_wedge_sign_from_residuals)(const int16_t *ds, const uint8_t *m, int N, int64_t limit);
uint64_t(*eb_compute_cdef_dist)(const uint16_t *dst, int32_t dstride, const uint16_t *src, const CdefList *dlist, int32_t cdef_count, BlockSize bsize, int32_t coeff_shift, int32_t pli);
uint64_t(*eb_compute_cdef_dist_8bit)(const uint8_t *dst8, int32_t dstride, const uint8_t *src8, const CdefList *dlist, int32_t cdef_count, BlockSize bsize, int32_t coeff_shift, int32_t pli);
void(*eb_av1_compute_stats)(int32_t wiener_win, const uint8_t *dgd8, const uint8_t *src8, int32_t h_start, int32_t h_end, int32_t v_start, int32_t v_end, int32_t dgd_stride, int32_t src_stride, int64_t *M, int64_t *H);
void(*eb_av1_compute_stats_highbd)(int32_t wiener_win, const uint8_t *dgd8, const uint8_t *src8, int32_t h_start, int32_t h_end, int32_t v_start, int32_t v_end, int32_t dgd_stride, int32_t src_stride, int64_t *M, int64_t *H, AomBitDepth bit_depth);
int64_t(*eb_av1_lowbd_pixel_proj_error)(const uint8_t *src8, int32_t width, int32_t height, int32_t src_stride, const uint8_t *dat8, int32_t dat_stride, int32_t *flt0, int32_t flt0_stride, int32_t *flt1, int32_t flt1_stride, int32_t xq[2], const SgrParamsType *params);
int64_t(*eb_av1_highbd_pixel_proj_error)(const uint8_t *src8, int32_t width, int32_t height, int32_t src_stride, const uint8_t *dat8, int32_t dat_stride, int32_t *flt0, int32_t flt0_stride, int32_t *flt1, int32_t flt1_stride, int32_t xq[2], const SgrParamsType *params);
void(*eb_subtract_average)(int16_t *pred_buf_q3, int32_t width, int32_t height, int32_t round_offset, int32_t num_pel_log2);
int64_t(*eb_av1_calc_frame_error)(const uint8_t *const ref, int stride, const uint8_t *const dst, int p_width, int p_height, int p_stride);
void(*eb_av1_fwd_txfm2d_4x16)(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
void(*eb_av1_fwd_txfm2d_16x4)(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
void(*eb_av1_fwd_txfm2d_4x8)(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
void(*eb_av1_fwd_txfm2d_8x4)(int16_t *input, int32_t *output, uint32_t inputStride, TxType transform_type, uint8_t  bit_depth);
void(*eb_av1_fwd_txfm2d_8x16)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
void(*eb_av1_fwd_txfm2d_16x8)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
void(*eb_av1_fwd_txfm2d_4x16)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
void(*eb_av1_fwd_txfm2d_16x4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
void(*eb_av1_fwd_txfm2d_4x8)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
void(*eb_av1_fwd_txfm2d_8x4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
void(*eb_av1_fwd_txfm2d_32x16)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
void(*eb_av1_fwd_txfm2d_32x8)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
void(*eb_av1_fwd_txfm2d_8x32)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
void(*eb_av1_fwd_txfm2d_16x32)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
void(*eb_av1_fwd_txfm2d_32x64)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
void(*eb_av1_fwd_txfm2d_64x32)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
void(*eb_av1_fwd_txfm2d_16x64)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
void(*eb_av1_fwd_txfm2d_64x16)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
void(*eb_av1_fwd_txfm2d_64x64)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
void(*eb_av1_fwd_txfm2d_32x32)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
void(*eb_av1_fwd_txfm2d_16x16)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
void(*eb_av1_fwd_txfm2d_8x8)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
void(*eb_av1_fwd_txfm2d_4x4)(int16_t *input, int32_t *output, uint32_t input_stride, TxType transform_type, uint8_t  bit_depth);
void(*eb_smooth_v_predictor)(uint8_t *dst, ptrdiff_t stride, int32_t bw, int32_t bh, const uint8_t *above, const uint8_t *left);
void(*get_proj_subspace)(const uint8_t *src8, int width, int height, int src_stride, const uint8_t *dat8, int dat_stride, int use_highbitdepth, int32_t *flt0, int flt0_stride, int32_t *flt1, int flt1_stride, int *xq, const SgrParamsType *params);
uint64_t(*handle_transform16x64)(int32_t *output);
uint64_t(*handle_transform32x64)(int32_t *output);
uint64_t(*handle_transform64x16)(int32_t *output);
uint64_t(*handle_transform64x32)(int32_t *output);
uint64_t(*handle_transform64x64)(int32_t *output);
uint64_t(*search_one_dual)(int *lev0, int *lev1, int nb_strengths, uint64_t(**mse)[64], int sb_count, int fast, int start_gi, int end_gi);
uint32_t(*eb_aom_mse16x16)(const uint8_t *src_ptr, int32_t  source_stride, const uint8_t *ref_ptr, int32_t  recon_stride, uint32_t *sse);
void(*eb_aom_quantize_b)(const TranLow *coeff_ptr, intptr_t n_coeffs, int32_t skip_block, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan);
void(*eb_aom_quantize_b_32x32)(const TranLow *coeff_ptr, intptr_t n_coeffs, int32_t skip_block, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan);
void(*eb_aom_quantize_b_64x64)(const TranLow *coeff_ptr, intptr_t n_coeffs, int32_t skip_block, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan);
void(*eb_aom_highbd_quantize_b)(const TranLow *coeff_ptr, intptr_t n_coeffs, int32_t skip_block, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan);
void(*eb_aom_highbd_quantize_b_32x32)(const TranLow *coeff_ptr, intptr_t n_coeffs, int32_t skip_block, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan);
void(*eb_aom_highbd_quantize_b_64x64)(const TranLow *coeff_ptr, intptr_t n_coeffs, int32_t skip_block, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan);
void(*eb_av1_quantize_fp)(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan);
void(*eb_av1_highbd_quantize_fp)(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan, int16_t log_scale);
void(*eb_av1_quantize_fp_32x32)(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan);
void(*eb_av1_quantize_fp_64x64)(const TranLow *coeff_ptr, intptr_t n_coeffs, const int16_t *zbin_ptr, const int16_t *round_ptr, const int16_t *quant_ptr, const int16_t *quant_shift_ptr, TranLow *qcoeff_ptr, TranLow *dqcoeff_ptr, const int16_t *dequant_ptr, uint16_t *eob_ptr, const int16_t *scan, const int16_t *iscan);
void(*eb_aom_highbd_8_mse16x16)(const uint8_t *src_ptr, int32_t  source_stride, const uint8_t *ref_ptr, int32_t  recon_stride, uint32_t *sse);
uint32_t(*eb_aom_sad128x128)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
void(*eb_aom_sad128x128x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
uint32_t(*eb_aom_sad128x64)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
void(*eb_aom_sad128x64x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
uint32_t(*eb_aom_sad16x16)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
void(*eb_aom_sad16x16x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
uint32_t(*eb_aom_sad16x32)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
void(*eb_aom_sad16x32x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
uint32_t(*eb_aom_sad16x4)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
void(*eb_aom_sad16x4x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
uint32_t(*eb_aom_sad16x64)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
void(*eb_aom_sad16x64x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
uint32_t(*eb_aom_sad16x8)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
void(*eb_aom_sad16x8x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
uint32_t(*eb_aom_sad32x16)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
void(*eb_aom_sad32x16x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
uint32_t(*eb_aom_sad32x32)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
void(*eb_aom_sad32x32x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
uint32_t(*eb_aom_sad32x64)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
void(*eb_aom_sad32x64x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
uint32_t(*eb_aom_sad32x8)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
void(*eb_aom_sad32x8x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
uint32_t(*eb_aom_sad4x16)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
void(*eb_aom_sad4x16x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
uint32_t(*eb_aom_sad4x4)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
void(*eb_aom_sad4x4x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
uint32_t(*eb_aom_sad4x8)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
void(*eb_aom_sad4x8x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
uint32_t(*eb_aom_sad64x128)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
void(*eb_aom_sad64x128x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
uint32_t(*eb_aom_sad64x16)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
void(*eb_aom_sad64x16x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
uint32_t(*eb_aom_sad64x32)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
void(*eb_aom_sad64x32x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
uint32_t(*eb_aom_sad64x64)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
void(*eb_aom_sad64x64x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
uint32_t(*eb_aom_sad8x16)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
void(*eb_aom_sad8x16x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
uint32_t(*eb_aom_sad8x32)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
void(*eb_aom_sad8x32x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
uint32_t(*eb_aom_sad8x4)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
void(*eb_aom_sad8x4x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
uint32_t(*eb_aom_sad8x8)(const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr, int ref_stride);
void(*eb_aom_sad8x8x4d)(const uint8_t *src_ptr, int src_stride, const uint8_t * const ref_ptr[], int ref_stride, uint32_t *sad_array);
void(*aom_upsampled_pred) (MacroBlockD *xd, const struct AV1Common *const cm, int mi_row, int mi_col, const MV *const mv, uint8_t *comp_pred, int width, int height, int subpel_x_q3, int subpel_y_q3, const uint8_t *ref, int ref_stride, int subpel_search);
unsigned int(*aom_obmc_sad128x128)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
unsigned int(*aom_obmc_sad128x64)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
unsigned int(*aom_obmc_sad16x16)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
unsigned int(*aom_obmc_sad16x32)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
unsigned int(*aom_obmc_sad16x4)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
unsigned int(*aom_obmc_sad16x64)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
unsigned int(*aom_obmc_sad16x8)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
unsigned int(*aom_obmc_sad32x16)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
unsigned int(*aom_obmc_sad32x32)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
unsigned int(*aom_obmc_sad32x64)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
unsigned int(*aom_obmc_sad32x8)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
unsigned int(*aom_obmc_sad4x16)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
unsigned int(*aom_obmc_sad4x4)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
unsigned int(*aom_obmc_sad4x8)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
unsigned int(*aom_obmc_sad64x128)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
unsigned int(*aom_obmc_sad64x16)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
unsigned int(*aom_obmc_sad64x32)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
unsigned int(*aom_obmc_sad64x64)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
unsigned int(*aom_obmc_sad8x16)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
unsigned int(*aom_obmc_sad8x32)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
unsigned int(*aom_obmc_sad8x4)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
unsigned int(*aom_obmc_sad8x8)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask);
unsigned int(*aom_obmc_sub_pixel_variance128x128)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_sub_pixel_variance128x64)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_sub_pixel_variance16x16)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_sub_pixel_variance16x32)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_sub_pixel_variance16x4)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_sub_pixel_variance16x64)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_sub_pixel_variance16x8)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_sub_pixel_variance32x16)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_sub_pixel_variance32x32)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_sub_pixel_variance32x64)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_sub_pixel_variance32x8)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_sub_pixel_variance4x16)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_sub_pixel_variance4x4)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_sub_pixel_variance4x8)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_sub_pixel_variance64x128)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_sub_pixel_variance64x16)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_sub_pixel_variance64x32)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_sub_pixel_variance64x64)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_sub_pixel_variance8x16)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_sub_pixel_variance8x32)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_sub_pixel_variance8x4)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_sub_pixel_variance8x8)(const uint8_t *pre, int pre_stride, int xoffset, int yoffset, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_variance128x128)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_variance128x64)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_variance16x16)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_variance16x32)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_variance16x4)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_variance16x64)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_variance16x8)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_variance32x16)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_variance32x32)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_variance32x64)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_variance32x8)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_variance4x16)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_variance4x4)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_variance4x8)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_variance64x128)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_variance64x16)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_variance64x32)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_variance64x64)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_variance8x16)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_variance8x32)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_variance8x4)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*aom_obmc_variance8x8)(const uint8_t *pre, int pre_stride, const int32_t *wsrc, const int32_t *mask, unsigned int *sse);
unsigned int(*eb_aom_variance4x4)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_variance4x8)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_variance4x16)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_variance8x4)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_variance8x8)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_variance8x16)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_variance8x32)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_variance16x4)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_variance16x8)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_variance16x16)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_variance16x32)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_variance16x64)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_variance32x8)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_variance32x16)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_variance32x32)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_variance32x64)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_variance64x16)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_variance64x32)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_variance64x64)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_variance64x128)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_variance128x64)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_variance128x128)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_highbd_10_variance4x4)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_highbd_10_variance4x8)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_highbd_10_variance4x16)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_highbd_10_variance8x4)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_highbd_10_variance8x8)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_highbd_10_variance8x16)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_highbd_10_variance8x32)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_highbd_10_variance16x4)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_highbd_10_variance16x8)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_highbd_10_variance16x16)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_highbd_10_variance16x32)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_highbd_10_variance16x64)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_highbd_10_variance32x8)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_highbd_10_variance32x16)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_highbd_10_variance32x32)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_highbd_10_variance32x64)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_highbd_10_variance64x16)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_highbd_10_variance64x32)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_highbd_10_variance64x64)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_highbd_10_variance64x128)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_highbd_10_variance128x64)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
unsigned int(*eb_aom_highbd_10_variance128x128)(const uint8_t *src_ptr, int source_stride, const uint8_t *ref_ptr, int ref_stride, unsigned int *sse);
void(*eb_aom_ifft16x16_float)(const float *input, float *temp, float *output);
void(*eb_aom_ifft2x2_float)(const float *input, float *temp, float *output);
void(*eb_aom_ifft32x32_float)(const float *input, float *temp, float *output);
void(*eb_aom_ifft4x4_float)(const float *input, float *temp, float *output);
void(*eb_aom_ifft8x8_float)(const float *input, float *temp, float *output);
void(*eb_aom_fft16x16_float)(const float *input, float *temp, float *output);
void(*eb_aom_fft2x2_float)(const float *input, float *temp, float *output);
void(*eb_aom_fft32x32_float)(const float *input, float *temp, float *output);
void(*eb_aom_fft4x4_float)(const float *input, float *temp, float *output);
void(*eb_aom_fft8x8_float)(const float *input, float *temp, float *output);
void(*eb_av1_get_nz_map_contexts)(const uint8_t *const levels, const int16_t *const scan, const uint16_t eob, const TxSize tx_size, const TxClass tx_class, int8_t *const coeff_contexts);
void(*residual_kernel8bit)(uint8_t *input, uint32_t input_stride, uint8_t *pred, uint32_t pred_stride, int16_t *residual, uint32_t residual_stride, uint32_t area_width, uint32_t area_height);
void(*sad_loop_kernel)(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride, uint32_t block_height, uint32_t block_width, uint64_t *best_sad, int16_t *x_search_center, int16_t *y_search_center, uint32_t src_stride_raw, int16_t search_area_width, int16_t search_area_height);
void(*sad_loop_kernel_sparse)(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride, uint32_t block_height, uint32_t block_width, uint64_t *best_sad, int16_t *x_search_center, int16_t *y_search_center, uint32_t src_stride_raw, int16_t search_area_width, int16_t search_area_height);
void(*sad_loop_kernel_hme_l0)(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride, uint32_t block_height, uint32_t block_width, uint64_t *best_sad, int16_t *x_search_center, int16_t *y_search_center, uint32_t src_stride_raw, int16_t search_area_width, int16_t search_area_height);
void(*eb_av1_txb_init_levels)(const TranLow *const coeff, const int32_t width, const int32_t height, uint8_t *const levels);
void(*av1_get_gradient_hist)(const uint8_t *src, int src_stride, int rows, int cols, uint64_t *hist);
double(*av1_compute_cross_correlation)(unsigned char *im1, int stride1, int x1, int y1, unsigned char *im2, int stride2, int x2, int y2);
void(*av1_k_means_dim1)(const int* data, int* centroids, uint8_t* indices, int n, int k, int max_itr);
void(*av1_k_means_dim2)(const int* data, int* centroids, uint8_t* indices, int n, int k, int max_itr);
void(*av1_calc_indices_dim1)(const int* data, const int* centroids, uint8_t* indices, int n, int k);
void(*av1_calc_indices_dim2)(const int* data, const int* centroids, uint8_t* indices, int n, int k);
void(*svt_av1_apply_filtering)(const uint8_t *y_src, int y_src_stride, const uint8_t *y_pre, int y_pre_stride, const uint8_t *u_src, const uint8_t *v_src, int uv_src_stride, const uint8_t *u_pre, const uint8_t *v_pre, int uv_pre_stride, unsigned int block_width, unsigned int block_height, int ss_x, int ss_y, int strength, const int *blk_fw, int use_whole_blk, uint32_t *y_accum, uint16_t *y_count, uint32_t *u_accum, uint16_t *u_count, uint32_t *v_accum, uint16_t *v_count);
void(*svt_av1_apply_filtering_highbd)(const uint16_t *y_src, int y_src_stride, const uint16_t *y_pre, int y_pre_stride, const uint16_t *u_src, const uint16_t *v_src, int uv_src_stride, const uint16_t *u_pre, const uint16_t *v_pre, int uv_pre_stride, unsigned int block_width, unsigned int block_height, int ss_x, int ss_y, int strength, const int *blk_fw, int use_whole_blk, uint32_t *y_accum, uint16_t *y_count, uint32_t *u_accum, uint16_t *u_count, uint32_t *v_accum, uint16_t *v_count);
void(*svt_av1_apply_temporal_filter_planewise)(const uint8_t *y_src, int y_src_stride, const uint8_t *y_pre, int y_pre_stride, const uint8_t *u_src, const uint8_t *v_src, int uv_src_stride, const uint8_t *u_pre, const uint8_t *v_pre, int uv_pre_stride, unsigned int block_width, unsigned int block_height, int ss_x, int ss_y, const double *noise_levels, const int decay_control, uint32_t *y_accum, uint16_t *y_count, uint32_t *u_accum, uint16_t *u_count, uint32_t *v_accum, uint16_t *v_count);
uint32_t(*combined_averaging_ssd)(uint8_t *src, ptrdiff_t src_stride, uint8_t *ref1, ptrdiff_t ref1_stride, uint8_t *ref2, ptrdiff_t ref2_stride, uint32_t height, uint32_t width);
void(*ext_sad_calculation_8x8_16x16)(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride, uint32_t *p_best_sad_8x8, uint32_t *p_best_sad_16x16, uint32_t *p_best_mv8x8, uint32_t *p_best_mv16x16, uint32_t mv, uint32_t *p_sad16x16, uint32_t *p_sad8x8, EbBool sub_sad);
void(*ext_sad_calculation_32x32_64x64)(uint32_t *p_sad16x16, uint32_t *p_best_sad_32x32, uint32_t *p_best_sad_64x64, uint32_t *p_best_mv32x32, uint32_t *p_best_mv64x64, uint32_t mv, uint32_t *p_sad32x32);
void(*sad_calculation_8x8_16x16)(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride, uint32_t *p_best_sad_8x8, uint32_t *p_best_sad_16x16, uint32_t *p_best_mv8x8, uint32_t *p_best_mv16x16, uint32_t mv, uint32_t *p_sad16x16, EbBool sub_sad);
void(*sad_calculation_32x32_64x64)(uint32_t *p_sad16x16, uint32_t *p_best_sad_32x32, uint32_t *p_best_sad_64x64, uint32_t *p_best_mv32x32, uint32_t *p_best_mv64x64, uint32_t mv);
void(*ext_all_sad_calculation_8x8_16x16)(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride, uint32_t mv, uint32_t *p_best_sad_8x8, uint32_t *p_best_sad_16x16, uint32_t *p_best_mv8x8, uint32_t *p_best_mv16x16, uint32_t p_eight_sad16x16[16][8], uint32_t p_eight_sad8x8[64][8]);
void(*ext_eigth_sad_calculation_nsq)(uint32_t p_sad8x8[64][8], uint32_t p_sad16x16[16][8], uint32_t p_sad32x32[4][8], uint32_t *p_best_sad_64x32, uint32_t *p_best_mv64x32, uint32_t *p_best_sad_32x16, uint32_t *p_best_mv32x16, uint32_t *p_best_sad_16x8, uint32_t *p_best_mv16x8, uint32_t *p_best_sad_32x64, uint32_t *p_best_mv32x64, uint32_t *p_best_sad_16x32, uint32_t *p_best_mv16x32, uint32_t *p_best_sad_8x16, uint32_t *p_best_mv8x16, uint32_t *p_best_sad_32x8, uint32_t *p_best_mv32x8, uint32_t *p_best_sad_8x32, uint32_t *p_best_mv8x32, uint32_t *p_best_sad_64x16, uint32_t *p_best_mv64x16, uint32_t *p_best_sad_16x64, uint32_t *p_best_mv16x64, uint32_t mv);
void(*ext_eight_sad_calculation_32x32_64x64)(uint32_t p_sad16x16[16][8], uint32_t *p_best_sad_32x32, uint32_t *p_best_sad_64x64, uint32_t *p_best_mv32x32, uint32_t *p_best_mv64x64, uint32_t mv, uint32_t p_sad32x32[4][8]);
uint32_t(*eb_sad_kernel4x4)(const uint8_t *src, uint32_t src_stride, const uint8_t *ref, uint32_t ref_stride, uint32_t height, uint32_t width);
void(*get_eight_horizontal_search_point_results_8x8_16x16_pu)(uint8_t *src, uint32_t src_stride, uint8_t *ref, uint32_t ref_stride, uint32_t *p_best_sad_8x8, uint32_t *p_best_mv8x8, uint32_t *p_best_sad_16x16, uint32_t *p_best_mv16x16, uint32_t mv, uint16_t *p_sad16x16, EbBool sub_sad);
void(*get_eight_horizontal_search_point_results_32x32_64x64_pu)(uint16_t *p_sad16x16, uint32_t *p_best_sad_32x32, uint32_t *p_best_sad_64x64, uint32_t *p_best_mv32x32, uint32_t *p_best_mv64x64, uint32_t mv);
void(*initialize_buffer_32bits)(uint32_t* pointer, uint32_t count128, uint32_t count32, uint32_t value);
uint32_t(*nxm_sad_kernel_sub_sampled)(const uint8_t *src, uint32_t src_stride, const uint8_t *ref, uint32_t ref_stride, uint32_t height, uint32_t width);
uint32_t(*nxm_sad_kernel)(const uint8_t *src, uint32_t src_stride, const uint8_t *ref, uint32_t ref_stride, uint32_t height, uint32_t width);
uint32_t(*nxm_sad_avg_kernel)(uint8_t *src, uint32_t src_stride, uint8_t *ref1, uint32_t ref1_stride, uint8_t *ref2, uint32_t ref2_stride, uint32_t height, uint32_t width);
void(*avc_style_luma_interpolation_filter)(EbByte ref_pic, uint32_t src_stride, EbByte dst, uint32_t dst_stride, uint32_t pu_width, uint32_t pu_height, EbByte temp_buf, EbBool skip, uint32_t frac_pos, uint8_t fractional_position);
uint64_t(*compute_mean_8x8)(uint8_t *input_samples, uint32_t input_stride, uint32_t input_area_width, uint32_t input_area_height);
uint64_t(*compute_mean_square_values_8x8)(uint8_t *input_samples, uint32_t input_stride, uint32_t input_area_width, uint32_t input_area_height);
uint64_t(*compute_sub_mean_8x8)(uint8_t* input_samples, uint16_t input_stride);
void(*compute_interm_var_four8x8)(uint8_t *input_samples, uint16_t input_stride, uint64_t *mean_of8x8_blocks, uint64_t *mean_of_squared8x8_blocks);
uint32_t(*sad_16b_kernel)(uint16_t *src, uint32_t src_stride, uint16_t *ref, uint32_t ref_stride, uint32_t height, uint32_t width);
void(*pme_sad_loop_kernel)(uint8_t* src, uint32_t src_stride, uint8_t* ref, uint32_t ref_stride, uint32_t block_height, uint32_t block_width, uint32_t* best_sad, int16_t* best_mvx, int16_t* best_mvy, int16_t search_position_start_x, int16_t search_position_start_y, int16_t search_area_width, int16_t search_area_height, int16_t search_step, int16_t mvx, int16_t mvy);


/**************************************
 * Instruction Set Support
 **************************************/

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

void setup_rtcd_internal(CPU_FLAGS flags) {
    /** Should be done during library initialization,
        but for safe limiting cpu flags again. */
    (void)flags;
    //to use C: flags=0
    aom_sse = aom_sse_c;

    aom_highbd_sse = aom_highbd_sse_c;

    av1_wedge_compute_delta_squares = av1_wedge_compute_delta_squares_c;
    av1_wedge_sign_from_residuals = av1_wedge_sign_from_residuals_c;

    eb_compute_cdef_dist = compute_cdef_dist_c;
    eb_compute_cdef_dist_8bit = compute_cdef_dist_8bit_c;

    eb_av1_compute_stats = eb_av1_compute_stats_c;
    eb_av1_compute_stats_highbd = eb_av1_compute_stats_highbd_c;

    eb_av1_lowbd_pixel_proj_error = eb_av1_lowbd_pixel_proj_error_c;
    eb_av1_highbd_pixel_proj_error = eb_av1_highbd_pixel_proj_error_c;
    eb_av1_calc_frame_error = eb_av1_calc_frame_error_c;

    eb_subtract_average = eb_subtract_average_c;

    get_proj_subspace = get_proj_subspace_c;

    search_one_dual = search_one_dual_c;

    eb_aom_mse16x16 = eb_aom_mse16x16_c;

    eb_aom_quantize_b = eb_aom_quantize_b_c_ii;

    eb_aom_quantize_b_32x32 = eb_aom_quantize_b_32x32_c_ii;

    eb_aom_highbd_quantize_b_32x32 = eb_aom_highbd_quantize_b_32x32_c;

    eb_aom_highbd_quantize_b = eb_aom_highbd_quantize_b_c;

    eb_av1_quantize_fp = eb_av1_quantize_fp_c;

    eb_av1_quantize_fp_32x32 = eb_av1_quantize_fp_32x32_c;

    eb_av1_quantize_fp_64x64 = eb_av1_quantize_fp_64x64_c;

    eb_av1_highbd_quantize_fp = eb_av1_highbd_quantize_fp_c;

    eb_aom_highbd_8_mse16x16 = eb_aom_highbd_8_mse16x16_c;


    //SAD
    eb_aom_sad4x4 = eb_aom_sad4x4_c;
    eb_aom_sad4x4x4d = eb_aom_sad4x4x4d_c;
    eb_aom_sad4x16 = eb_aom_sad4x16_c;
    eb_aom_sad4x16x4d = eb_aom_sad4x16x4d_c;
    eb_aom_sad4x8 = eb_aom_sad4x8_c;
    eb_aom_sad4x8x4d = eb_aom_sad4x8x4d_c;
    eb_aom_sad64x128x4d = eb_aom_sad64x128x4d_c;
    eb_aom_sad64x16x4d = eb_aom_sad64x16x4d_c;
    eb_aom_sad64x32x4d = eb_aom_sad64x32x4d_c;
    eb_aom_sad64x64x4d = eb_aom_sad64x64x4d_c;
    eb_aom_sad8x16 = eb_aom_sad8x16_c;
    eb_aom_sad8x16x4d = eb_aom_sad8x16x4d_c;
    eb_aom_sad8x32 = eb_aom_sad8x32_c;
    eb_aom_sad8x32x4d = eb_aom_sad8x32x4d_c;
    eb_aom_sad8x8 = eb_aom_sad8x8_c;
    eb_aom_sad8x8x4d = eb_aom_sad8x8x4d_c;
    eb_aom_sad16x4 = eb_aom_sad16x4_c;
    eb_aom_sad16x4x4d = eb_aom_sad16x4x4d_c;
    eb_aom_sad32x8 = eb_aom_sad32x8_c;
    eb_aom_sad32x8x4d = eb_aom_sad32x8x4d_c;
    eb_aom_sad16x64 = eb_aom_sad16x64_c;
    eb_aom_sad16x64x4d = eb_aom_sad16x64x4d_c;
    eb_aom_sad32x16 = eb_aom_sad32x16_c;
    eb_aom_sad32x16x4d = eb_aom_sad32x16x4d_c;
    eb_aom_sad16x32 = eb_aom_sad16x32_c;
    eb_aom_sad16x32x4d = eb_aom_sad16x32x4d_c;
    eb_aom_sad32x64 = eb_aom_sad32x64_c;
    eb_aom_sad32x64x4d = eb_aom_sad32x64x4d_c;
    eb_aom_sad32x32 = eb_aom_sad32x32_c;
    eb_aom_sad32x32x4d = eb_aom_sad32x32x4d_c;
    eb_aom_sad16x16 = eb_aom_sad16x16_c;
    eb_aom_sad16x16x4d = eb_aom_sad16x16x4d_c;
    eb_aom_sad16x8 = eb_aom_sad16x8_c;
    eb_aom_sad16x8x4d = eb_aom_sad16x8x4d_c;
    eb_aom_sad8x4 = eb_aom_sad8x4_c;
    eb_aom_sad8x4x4d = eb_aom_sad8x4x4d_c;

    eb_aom_sad64x128 = eb_aom_sad64x128_c;
    eb_aom_sad64x16 = eb_aom_sad64x16_c;
    eb_aom_sad64x32 = eb_aom_sad64x32_c;
    eb_aom_sad64x64 = eb_aom_sad64x64_c;
    eb_aom_sad128x128 = eb_aom_sad128x128_c;
    eb_aom_sad128x128x4d = eb_aom_sad128x128x4d_c;
    eb_aom_sad128x64 = eb_aom_sad128x64_c;
    eb_aom_sad128x64x4d = eb_aom_sad128x64x4d_c;
    eb_av1_txb_init_levels = eb_av1_txb_init_levels_c;

    aom_upsampled_pred = aom_upsampled_pred_c;

    aom_obmc_sad128x128 = aom_obmc_sad128x128_c;
    aom_obmc_sad128x64 = aom_obmc_sad128x64_c;
    aom_obmc_sad16x16 = aom_obmc_sad16x16_c;
    aom_obmc_sad16x32 = aom_obmc_sad16x32_c;
    aom_obmc_sad16x4 = aom_obmc_sad16x4_c;
    aom_obmc_sad16x64 = aom_obmc_sad16x64_c;
    aom_obmc_sad16x8 = aom_obmc_sad16x8_c;
    aom_obmc_sad32x16 = aom_obmc_sad32x16_c;
    aom_obmc_sad32x32 = aom_obmc_sad32x32_c;
    aom_obmc_sad32x64 = aom_obmc_sad32x64_c;
    aom_obmc_sad32x8 = aom_obmc_sad32x8_c;
    aom_obmc_sad4x16 = aom_obmc_sad4x16_c;
    aom_obmc_sad4x4 = aom_obmc_sad4x4_c;
    aom_obmc_sad4x8 = aom_obmc_sad4x8_c;
    aom_obmc_sad64x128 = aom_obmc_sad64x128_c;
    aom_obmc_sad64x16 = aom_obmc_sad64x16_c;
    aom_obmc_sad64x32 = aom_obmc_sad64x32_c;
    aom_obmc_sad64x64 = aom_obmc_sad64x64_c;
    aom_obmc_sad8x16 = aom_obmc_sad8x16_c;
    aom_obmc_sad8x32 = aom_obmc_sad8x32_c;
    aom_obmc_sad8x4 = aom_obmc_sad8x4_c;
    aom_obmc_sad8x8 = aom_obmc_sad8x8_c;
    aom_obmc_sub_pixel_variance128x128 = aom_obmc_sub_pixel_variance128x128_c;
    aom_obmc_sub_pixel_variance128x64 = aom_obmc_sub_pixel_variance128x64_c;
    aom_obmc_sub_pixel_variance16x16 = aom_obmc_sub_pixel_variance16x16_c;
    aom_obmc_sub_pixel_variance16x32 = aom_obmc_sub_pixel_variance16x32_c;
    aom_obmc_sub_pixel_variance16x4 = aom_obmc_sub_pixel_variance16x4_c;
    aom_obmc_sub_pixel_variance16x64 = aom_obmc_sub_pixel_variance16x64_c;
    aom_obmc_sub_pixel_variance16x8 = aom_obmc_sub_pixel_variance16x8_c;
    aom_obmc_sub_pixel_variance32x16 = aom_obmc_sub_pixel_variance32x16_c;
    aom_obmc_sub_pixel_variance32x32 = aom_obmc_sub_pixel_variance32x32_c;
    aom_obmc_sub_pixel_variance32x64 = aom_obmc_sub_pixel_variance32x64_c;
    aom_obmc_sub_pixel_variance32x8 = aom_obmc_sub_pixel_variance32x8_c;
    aom_obmc_sub_pixel_variance4x16 = aom_obmc_sub_pixel_variance4x16_c;
    aom_obmc_sub_pixel_variance4x4 = aom_obmc_sub_pixel_variance4x4_c;
    aom_obmc_sub_pixel_variance4x8 = aom_obmc_sub_pixel_variance4x8_c;
    aom_obmc_sub_pixel_variance64x128 = aom_obmc_sub_pixel_variance64x128_c;
    aom_obmc_sub_pixel_variance64x16 = aom_obmc_sub_pixel_variance64x16_c;
    aom_obmc_sub_pixel_variance64x32 = aom_obmc_sub_pixel_variance64x32_c;
    aom_obmc_sub_pixel_variance64x64 = aom_obmc_sub_pixel_variance64x64_c;
    aom_obmc_sub_pixel_variance8x16 = aom_obmc_sub_pixel_variance8x16_c;
    aom_obmc_sub_pixel_variance8x32 = aom_obmc_sub_pixel_variance8x32_c;
    aom_obmc_sub_pixel_variance8x4 = aom_obmc_sub_pixel_variance8x4_c;
    aom_obmc_sub_pixel_variance8x8 = aom_obmc_sub_pixel_variance8x8_c;
    aom_obmc_variance128x128 = aom_obmc_variance128x128_c;
    aom_obmc_variance128x64 = aom_obmc_variance128x64_c;
    aom_obmc_variance16x16 = aom_obmc_variance16x16_c;
    aom_obmc_variance16x32 = aom_obmc_variance16x32_c;
    aom_obmc_variance16x4 = aom_obmc_variance16x4_c;
    aom_obmc_variance16x64 = aom_obmc_variance16x64_c;
    aom_obmc_variance16x8 = aom_obmc_variance16x8_c;
    aom_obmc_variance32x16 = aom_obmc_variance32x16_c;
    aom_obmc_variance32x32 = aom_obmc_variance32x32_c;
    aom_obmc_variance32x64 = aom_obmc_variance32x64_c;
    aom_obmc_variance32x8 = aom_obmc_variance32x8_c;
    aom_obmc_variance4x16 = aom_obmc_variance4x16_c;
    aom_obmc_variance4x4 = aom_obmc_variance4x4_c;
    aom_obmc_variance4x8 = aom_obmc_variance4x8_c;
    aom_obmc_variance64x128 = aom_obmc_variance64x128_c;
    aom_obmc_variance64x16 = aom_obmc_variance64x16_c;
    aom_obmc_variance64x32 = aom_obmc_variance64x32_c;
    aom_obmc_variance64x64 = aom_obmc_variance64x64_c;
    aom_obmc_variance8x16 = aom_obmc_variance8x16_c;
    aom_obmc_variance8x32 = aom_obmc_variance8x32_c;
    aom_obmc_variance8x4 = aom_obmc_variance8x4_c;
    aom_obmc_variance8x8 = aom_obmc_variance8x8_c;

    //VARIANCE
    eb_aom_variance4x4 = eb_aom_variance4x4_c;
    eb_aom_variance4x8 = eb_aom_variance4x8_c;
    eb_aom_variance4x16 = eb_aom_variance4x16_c;
    eb_aom_variance8x4 = eb_aom_variance8x4_c;
    eb_aom_variance8x8 = eb_aom_variance8x8_c;
    eb_aom_variance8x16 = eb_aom_variance8x16_c;
    eb_aom_variance8x32 = eb_aom_variance8x32_c;
    eb_aom_variance16x4 = eb_aom_variance16x4_c;
    eb_aom_variance16x8 = eb_aom_variance16x8_c;
    eb_aom_variance16x16 = eb_aom_variance16x16_c;
    eb_aom_variance16x32 = eb_aom_variance16x32_c;
    eb_aom_variance16x64 = eb_aom_variance16x64_c;
    eb_aom_variance32x8 = eb_aom_variance32x8_c;
    eb_aom_variance32x16 = eb_aom_variance32x16_c;
    eb_aom_variance32x32 = eb_aom_variance32x32_c;
    eb_aom_variance32x64 = eb_aom_variance32x64_c;
    eb_aom_variance64x16 = eb_aom_variance64x16_c;
    eb_aom_variance64x32 = eb_aom_variance64x32_c;
    eb_aom_variance64x64 = eb_aom_variance64x64_c;
    eb_aom_variance64x128 = eb_aom_variance64x128_c;
    eb_aom_variance128x64 = eb_aom_variance128x64_c;
    eb_aom_variance128x128 = eb_aom_variance128x128_c;

    // VARIANCE HBP
    eb_aom_highbd_10_variance4x4 = eb_aom_highbd_10_variance4x4_c;
    eb_aom_highbd_10_variance4x8 = eb_aom_highbd_10_variance4x8_c;
    eb_aom_highbd_10_variance4x16 = eb_aom_highbd_10_variance4x16_c;
    eb_aom_highbd_10_variance8x4 = eb_aom_highbd_10_variance8x4_c;
    eb_aom_highbd_10_variance8x8 = eb_aom_highbd_10_variance8x8_c;
    eb_aom_highbd_10_variance8x16 = eb_aom_highbd_10_variance8x16_c;
    eb_aom_highbd_10_variance8x32 = eb_aom_highbd_10_variance8x32_c;
    eb_aom_highbd_10_variance16x4 = eb_aom_highbd_10_variance16x4_c;
    eb_aom_highbd_10_variance16x8 = eb_aom_highbd_10_variance16x8_c;
    eb_aom_highbd_10_variance16x16 = eb_aom_highbd_10_variance16x16_c;
    eb_aom_highbd_10_variance16x32 = eb_aom_highbd_10_variance16x32_c;
    eb_aom_highbd_10_variance16x64 = eb_aom_highbd_10_variance16x64_c;
    eb_aom_highbd_10_variance32x8 = eb_aom_highbd_10_variance32x8_c;
    eb_aom_highbd_10_variance32x16 = eb_aom_highbd_10_variance32x16_c;
    eb_aom_highbd_10_variance32x32 = eb_aom_highbd_10_variance32x32_c;
    eb_aom_highbd_10_variance32x64 = eb_aom_highbd_10_variance32x64_c;
    eb_aom_highbd_10_variance64x16 = eb_aom_highbd_10_variance64x16_c;
    eb_aom_highbd_10_variance64x32 = eb_aom_highbd_10_variance64x32_c;
    eb_aom_highbd_10_variance64x64 = eb_aom_highbd_10_variance64x64_c;
    eb_aom_highbd_10_variance64x128 = eb_aom_highbd_10_variance64x128_c;
    eb_aom_highbd_10_variance128x64 = eb_aom_highbd_10_variance128x64_c;
    eb_aom_highbd_10_variance128x128 = eb_aom_highbd_10_variance128x128_c;

    //QIQ
    eb_aom_quantize_b_64x64 = eb_aom_quantize_b_64x64_c_ii;

    eb_aom_highbd_quantize_b_64x64 = eb_aom_highbd_quantize_b_64x64_c;
    // transform
    eb_av1_fwd_txfm2d_16x8 = eb_av1_fwd_txfm2d_16x8_c;
    eb_av1_fwd_txfm2d_8x16 = eb_av1_fwd_txfm2d_8x16_c;

    eb_av1_fwd_txfm2d_16x4 = eb_av1_fwd_txfm2d_16x4_c;
    eb_av1_fwd_txfm2d_4x16 = eb_av1_fwd_txfm2d_4x16_c;

    eb_av1_fwd_txfm2d_8x4 = eb_av1_fwd_txfm2d_8x4_c;
    eb_av1_fwd_txfm2d_4x8 = eb_av1_fwd_txfm2d_4x8_c;

    eb_av1_fwd_txfm2d_32x16 = eb_av1_fwd_txfm2d_32x16_c;
    eb_av1_fwd_txfm2d_32x8 = eb_av1_fwd_txfm2d_32x8_c;
    eb_av1_fwd_txfm2d_8x32 = eb_av1_fwd_txfm2d_8x32_c;
    eb_av1_fwd_txfm2d_16x32 = eb_av1_fwd_txfm2d_16x32_c;
    eb_av1_fwd_txfm2d_32x64 = eb_av1_fwd_txfm2d_32x64_c;
    eb_av1_fwd_txfm2d_64x32 = eb_av1_fwd_txfm2d_64x32_c;
    eb_av1_fwd_txfm2d_16x64 = eb_av1_fwd_txfm2d_16x64_c;
    eb_av1_fwd_txfm2d_64x16 = eb_av1_fwd_txfm2d_64x16_c;
    eb_av1_fwd_txfm2d_64x64 = av1_transform_two_d_64x64_c;
    eb_av1_fwd_txfm2d_32x32 = av1_transform_two_d_32x32_c;
    eb_av1_fwd_txfm2d_16x16 = av1_transform_two_d_16x16_c;

    eb_av1_fwd_txfm2d_8x8 = av1_transform_two_d_8x8_c;
    eb_av1_fwd_txfm2d_4x4 = av1_transform_two_d_4x4_c;

    handle_transform16x64 = handle_transform16x64_c;
    handle_transform32x64 = handle_transform32x64_c;
    handle_transform64x16 = handle_transform64x16_c;
    handle_transform64x32 = handle_transform64x32_c;
    handle_transform64x64 = handle_transform64x64_c;

    eb_aom_fft2x2_float = eb_aom_fft2x2_float_c;
    eb_aom_fft4x4_float = eb_aom_fft4x4_float_c;
    eb_aom_fft16x16_float = eb_aom_fft16x16_float_c;
    eb_aom_fft32x32_float = eb_aom_fft32x32_float_c;
    eb_aom_fft8x8_float = eb_aom_fft8x8_float_c;

    eb_aom_ifft16x16_float = eb_aom_ifft16x16_float_c;
    eb_aom_ifft32x32_float = eb_aom_ifft32x32_float_c;
    eb_aom_ifft8x8_float = eb_aom_ifft8x8_float_c;
    eb_aom_ifft2x2_float = eb_aom_ifft2x2_float_c;
    eb_aom_ifft4x4_float = eb_aom_ifft4x4_float_c;
    av1_get_gradient_hist = av1_get_gradient_hist_c;

    search_one_dual = search_one_dual_c;
    sad_loop_kernel_sparse = sad_loop_kernel_sparse_c;
    sad_loop_kernel = sad_loop_kernel_c;
    sad_loop_kernel_hme_l0 = sad_loop_kernel_c;

    svt_av1_apply_filtering = svt_av1_apply_filtering_c;
    svt_av1_apply_temporal_filter_planewise = svt_av1_apply_temporal_filter_planewise_c;
    svt_av1_apply_filtering_highbd = svt_av1_apply_filtering_highbd_c;
    combined_averaging_ssd = combined_averaging_ssd_c;
    ext_sad_calculation_8x8_16x16 = ext_sad_calculation_8x8_16x16_c;
    ext_sad_calculation_32x32_64x64 = ext_sad_calculation_32x32_64x64_c;
    sad_calculation_8x8_16x16 = sad_calculation_8x8_16x16_c;
    sad_calculation_32x32_64x64 = sad_calculation_32x32_64x64_c;
    ext_all_sad_calculation_8x8_16x16 = ext_all_sad_calculation_8x8_16x16_c;
    ext_eigth_sad_calculation_nsq = ext_eigth_sad_calculation_nsq_c;
    ext_eight_sad_calculation_32x32_64x64 = ext_eight_sad_calculation_32x32_64x64_c;
    eb_sad_kernel4x4 = fast_loop_nxm_sad_kernel;
    get_eight_horizontal_search_point_results_8x8_16x16_pu = get_eight_horizontal_search_point_results_8x8_16x16_pu_c;
    get_eight_horizontal_search_point_results_32x32_64x64_pu = get_eight_horizontal_search_point_results_32x32_64x64_pu_c;

    initialize_buffer_32bits = initialize_buffer_32bits_c;
    nxm_sad_kernel_sub_sampled = nxm_sad_kernel_helper_c;
    nxm_sad_kernel = nxm_sad_kernel_helper_c;
    nxm_sad_avg_kernel = nxm_sad_avg_kernel_helper_c;

    compute_mean_8x8 = compute_mean_c;
    compute_mean_square_values_8x8 = compute_mean_squared_values_c;
    compute_sub_mean_8x8 = compute_sub_mean_8x8_c;
    compute_interm_var_four8x8 = compute_interm_var_four8x8_c;
    sad_16b_kernel = sad_16b_kernel_c;
    av1_compute_cross_correlation = av1_compute_cross_correlation_c;
    av1_k_means_dim1 = av1_k_means_dim1_c;
    av1_k_means_dim2 = av1_k_means_dim2_c;
    av1_calc_indices_dim1 = av1_calc_indices_dim1_c;
    av1_calc_indices_dim2 = av1_calc_indices_dim2_c;

    eb_av1_get_nz_map_contexts = eb_av1_get_nz_map_contexts_c;

#if RESTRUCTURE_SAD
    pme_sad_loop_kernel = pme_sad_loop_kernel_c;
#endif

#ifdef ARCH_X86
    flags &= get_cpu_flags_to_use();
    if (flags & HAS_AVX2) aom_sse = aom_sse_avx2;
    if (flags & HAS_AVX2) aom_highbd_sse = aom_highbd_sse_avx2;
    if (flags & HAS_AVX2) av1_wedge_compute_delta_squares = av1_wedge_compute_delta_squares_avx2;
    if (flags & HAS_AVX2) av1_wedge_sign_from_residuals = av1_wedge_sign_from_residuals_avx2;
    if (flags & HAS_AVX2) eb_compute_cdef_dist = compute_cdef_dist_avx2;
    if (flags & HAS_AVX2) eb_compute_cdef_dist_8bit = compute_cdef_dist_8bit_avx2;
    if (flags & HAS_AVX2) eb_av1_compute_stats = eb_av1_compute_stats_avx2;
    if (flags & HAS_AVX2) eb_av1_compute_stats_highbd = eb_av1_compute_stats_highbd_avx2;
#ifndef NON_AVX512_SUPPORT
    if (flags & HAS_AVX512F) {
        eb_av1_compute_stats = eb_av1_compute_stats_avx512;
        eb_av1_compute_stats_highbd = eb_av1_compute_stats_highbd_avx512;
    }
#endif
        if (flags & HAS_AVX2) eb_av1_highbd_pixel_proj_error = eb_av1_highbd_pixel_proj_error_avx2;
        if (flags & HAS_AVX2) eb_av1_calc_frame_error = eb_av1_calc_frame_error_avx2;
        if (flags & HAS_AVX2) eb_subtract_average = eb_subtract_average_avx2;
        if (flags & HAS_AVX2) get_proj_subspace = get_proj_subspace_avx2;
        if (flags & HAS_AVX2) search_one_dual = search_one_dual_avx2;
        if (flags & HAS_AVX2) eb_aom_mse16x16 = eb_aom_mse16x16_avx2;
        if (flags & HAS_AVX2) eb_aom_quantize_b = eb_aom_quantize_b_avx2;
        if (flags & HAS_AVX2) eb_aom_quantize_b_32x32 = eb_aom_quantize_b_32x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_quantize_b_32x32 = eb_aom_highbd_quantize_b_32x32_avx2;
        if (flags & HAS_AVX2) eb_aom_highbd_quantize_b = eb_aom_highbd_quantize_b_avx2;
        if (flags & HAS_AVX2) eb_av1_lowbd_pixel_proj_error = eb_av1_lowbd_pixel_proj_error_avx2;
#ifndef NON_AVX512_SUPPORT
        if (flags & HAS_AVX512F) {
            eb_av1_lowbd_pixel_proj_error = eb_av1_lowbd_pixel_proj_error_avx512;
        }
#endif
            if (flags & HAS_AVX2) eb_av1_quantize_fp = eb_av1_quantize_fp_avx2;
            if (flags & HAS_AVX2) eb_av1_quantize_fp_32x32 = eb_av1_quantize_fp_32x32_avx2;
            if (flags & HAS_AVX2) eb_av1_quantize_fp_64x64 = eb_av1_quantize_fp_64x64_avx2;
            if (flags & HAS_AVX2) eb_av1_highbd_quantize_fp = eb_av1_highbd_quantize_fp_avx2;
            if (flags & HAS_SSE2) eb_aom_highbd_8_mse16x16 = eb_aom_highbd_8_mse16x16_sse2;
            if (flags & HAS_AVX2) eb_aom_sad4x4 = eb_aom_sad4x4_avx2;
            if (flags & HAS_AVX2) eb_aom_sad4x4x4d = eb_aom_sad4x4x4d_avx2;
            if (flags & HAS_AVX2) eb_aom_sad4x16 = eb_aom_sad4x16_avx2;
            if (flags & HAS_AVX2) eb_aom_sad4x16x4d = eb_aom_sad4x16x4d_avx2;
            if (flags & HAS_AVX2) eb_aom_sad4x8 = eb_aom_sad4x8_avx2;
            if (flags & HAS_AVX2) eb_aom_sad4x8x4d = eb_aom_sad4x8x4d_avx2;
            if (flags & HAS_AVX2) eb_aom_sad64x128x4d = eb_aom_sad64x128x4d_avx2;
            if (flags & HAS_AVX2) eb_aom_sad64x16x4d = eb_aom_sad64x16x4d_avx2;
            if (flags & HAS_AVX2) eb_aom_sad64x32x4d = eb_aom_sad64x32x4d_avx2;
            if (flags & HAS_AVX2) eb_aom_sad64x64x4d = eb_aom_sad64x64x4d_avx2;
            if (flags & HAS_AVX2) eb_aom_sad8x16 = eb_aom_sad8x16_avx2;
            if (flags & HAS_AVX2) eb_aom_sad8x16x4d = eb_aom_sad8x16x4d_avx2;
            if (flags & HAS_AVX2) eb_aom_sad8x32 = eb_aom_sad8x32_avx2;
            if (flags & HAS_AVX2) eb_aom_sad8x32x4d = eb_aom_sad8x32x4d_avx2;
            if (flags & HAS_AVX2) eb_aom_sad8x8 = eb_aom_sad8x8_avx2;
            if (flags & HAS_AVX2) eb_aom_sad8x8x4d = eb_aom_sad8x8x4d_avx2;
            if (flags & HAS_AVX2) eb_aom_sad16x4 = eb_aom_sad16x4_avx2;
            if (flags & HAS_AVX2) eb_aom_sad16x4x4d = eb_aom_sad16x4x4d_avx2;
            if (flags & HAS_AVX2) eb_aom_sad32x8 = eb_aom_sad32x8_avx2;
            if (flags & HAS_AVX2) eb_aom_sad32x8x4d = eb_aom_sad32x8x4d_avx2;
            if (flags & HAS_AVX2) eb_aom_sad16x64 = eb_aom_sad16x64_avx2;
            if (flags & HAS_AVX2) eb_aom_sad16x64x4d = eb_aom_sad16x64x4d_avx2;
            if (flags & HAS_AVX2) eb_aom_sad32x16 = eb_aom_sad32x16_avx2;
            if (flags & HAS_AVX2) eb_aom_sad32x16x4d = eb_aom_sad32x16x4d_avx2;
            if (flags & HAS_AVX2) eb_aom_sad16x32 = eb_aom_sad16x32_avx2;
            if (flags & HAS_AVX2) eb_aom_sad16x32x4d = eb_aom_sad16x32x4d_avx2;
            if (flags & HAS_AVX2) eb_aom_sad32x64 = eb_aom_sad32x64_avx2;
            if (flags & HAS_AVX2) eb_aom_sad32x64x4d = eb_aom_sad32x64x4d_avx2;
            if (flags & HAS_AVX2) eb_aom_sad32x32 = eb_aom_sad32x32_avx2;
            if (flags & HAS_AVX2) eb_aom_sad32x32x4d = eb_aom_sad32x32x4d_avx2;
            if (flags & HAS_AVX2) eb_aom_sad16x16 = eb_aom_sad16x16_avx2;
            if (flags & HAS_AVX2) eb_aom_sad16x16x4d = eb_aom_sad16x16x4d_avx2;
            if (flags & HAS_AVX2) eb_aom_sad16x8 = eb_aom_sad16x8_avx2;
            if (flags & HAS_AVX2) eb_aom_sad16x8x4d = eb_aom_sad16x8x4d_avx2;
            if (flags & HAS_AVX2) eb_aom_sad8x4 = eb_aom_sad8x4_avx2;
            if (flags & HAS_AVX2) eb_aom_sad8x4x4d = eb_aom_sad8x4x4d_avx2;
            if (flags & HAS_AVX2) eb_aom_sad64x128 = eb_aom_sad64x128_avx2;
            if (flags & HAS_AVX2) eb_aom_sad64x16 = eb_aom_sad64x16_avx2;
            if (flags & HAS_AVX2) eb_aom_sad64x32 = eb_aom_sad64x32_avx2;
            if (flags & HAS_AVX2) eb_aom_sad64x64 = eb_aom_sad64x64_avx2;
            if (flags & HAS_AVX2) eb_aom_sad128x128 = eb_aom_sad128x128_avx2;
            if (flags & HAS_AVX2) eb_aom_sad128x128x4d = eb_aom_sad128x128x4d_avx2;
            if (flags & HAS_AVX2) eb_aom_sad128x64 = eb_aom_sad128x64_avx2;
            if (flags & HAS_AVX2) eb_aom_sad128x64x4d = eb_aom_sad128x64x4d_avx2;
            if (flags & HAS_AVX2) eb_av1_txb_init_levels = eb_av1_txb_init_levels_avx2;
#ifndef NON_AVX512_SUPPORT
            if (flags & HAS_AVX512F) {
                eb_aom_sad64x128 = eb_aom_sad64x128_avx512;
                eb_aom_sad64x16 = eb_aom_sad64x16_avx512;
                eb_aom_sad64x32 = eb_aom_sad64x32_avx512;
                eb_aom_sad64x64 = eb_aom_sad64x64_avx512;
                eb_aom_sad128x128 = eb_aom_sad128x128_avx512;
                eb_aom_sad128x128x4d = eb_aom_sad128x128x4d_avx512;
                eb_aom_sad128x64 = eb_aom_sad128x64_avx512;
                eb_aom_sad128x64x4d = eb_aom_sad128x64x4d_avx512;
                eb_av1_txb_init_levels = eb_av1_txb_init_levels_avx512;
            }
#endif // !NON_AVX512_SUPPORT
                if (flags & HAS_AVX2) aom_upsampled_pred = aom_upsampled_pred_sse2;
                if (flags & HAS_AVX2) aom_obmc_sad128x128 = aom_obmc_sad128x128_avx2;
                if (flags & HAS_AVX2) aom_obmc_sad128x64 = aom_obmc_sad128x64_avx2;
                if (flags & HAS_AVX2) aom_obmc_sad16x16 = aom_obmc_sad16x16_avx2;
                if (flags & HAS_AVX2) aom_obmc_sad16x32 = aom_obmc_sad16x32_avx2;
                if (flags & HAS_AVX2) aom_obmc_sad16x4 = aom_obmc_sad16x4_avx2;
                if (flags & HAS_AVX2) aom_obmc_sad16x64 = aom_obmc_sad16x64_avx2;
                if (flags & HAS_AVX2) aom_obmc_sad16x8 = aom_obmc_sad16x8_avx2;
                if (flags & HAS_AVX2) aom_obmc_sad32x16 = aom_obmc_sad32x16_avx2;
                if (flags & HAS_AVX2) aom_obmc_sad32x32 = aom_obmc_sad32x32_avx2;
                if (flags & HAS_AVX2) aom_obmc_sad32x64 = aom_obmc_sad32x64_avx2;
                if (flags & HAS_AVX2) aom_obmc_sad32x8 = aom_obmc_sad32x8_avx2;
                if (flags & HAS_AVX2) aom_obmc_sad4x16 = aom_obmc_sad4x16_avx2;
                if (flags & HAS_AVX2) aom_obmc_sad4x4 = aom_obmc_sad4x4_avx2;
                if (flags & HAS_AVX2) aom_obmc_sad4x8 = aom_obmc_sad4x8_avx2;
                if (flags & HAS_AVX2) aom_obmc_sad64x128 = aom_obmc_sad64x128_avx2;
                if (flags & HAS_AVX2) aom_obmc_sad64x16 = aom_obmc_sad64x16_avx2;
                if (flags & HAS_AVX2) aom_obmc_sad64x32 = aom_obmc_sad64x32_avx2;
                if (flags & HAS_AVX2) aom_obmc_sad64x64 = aom_obmc_sad64x64_avx2;
                if (flags & HAS_AVX2) aom_obmc_sad8x16 = aom_obmc_sad8x16_avx2;
                if (flags & HAS_AVX2) aom_obmc_sad8x32 = aom_obmc_sad8x32_avx2;
                if (flags & HAS_AVX2) aom_obmc_sad8x4 = aom_obmc_sad8x4_avx2;
                if (flags & HAS_AVX2) aom_obmc_sad8x8 = aom_obmc_sad8x8_avx2;
                if (flags & HAS_SSE4_1) aom_obmc_sub_pixel_variance128x128 = aom_obmc_sub_pixel_variance128x128_sse4_1;
                if (flags & HAS_SSE4_1) aom_obmc_sub_pixel_variance128x64 = aom_obmc_sub_pixel_variance128x64_sse4_1;
                if (flags & HAS_SSE4_1) aom_obmc_sub_pixel_variance16x16 = aom_obmc_sub_pixel_variance16x16_sse4_1;
                if (flags & HAS_SSE4_1) aom_obmc_sub_pixel_variance16x32 = aom_obmc_sub_pixel_variance16x32_sse4_1;
                if (flags & HAS_SSE4_1) aom_obmc_sub_pixel_variance16x4 = aom_obmc_sub_pixel_variance16x4_sse4_1;
                if (flags & HAS_SSE4_1) aom_obmc_sub_pixel_variance16x64 = aom_obmc_sub_pixel_variance16x64_sse4_1;
                if (flags & HAS_SSE4_1) aom_obmc_sub_pixel_variance16x8 = aom_obmc_sub_pixel_variance16x8_sse4_1;
                if (flags & HAS_SSE4_1) aom_obmc_sub_pixel_variance32x16 = aom_obmc_sub_pixel_variance32x16_sse4_1;
                if (flags & HAS_SSE4_1) aom_obmc_sub_pixel_variance32x32 = aom_obmc_sub_pixel_variance32x32_sse4_1;
                if (flags & HAS_SSE4_1) aom_obmc_sub_pixel_variance32x64 = aom_obmc_sub_pixel_variance32x64_sse4_1;
                if (flags & HAS_SSE4_1) aom_obmc_sub_pixel_variance32x8 = aom_obmc_sub_pixel_variance32x8_sse4_1;
                if (flags & HAS_SSE4_1) aom_obmc_sub_pixel_variance4x16 = aom_obmc_sub_pixel_variance4x16_sse4_1;
                if (flags & HAS_SSE4_1) aom_obmc_sub_pixel_variance4x4 = aom_obmc_sub_pixel_variance4x4_sse4_1;
                if (flags & HAS_SSE4_1) aom_obmc_sub_pixel_variance4x8 = aom_obmc_sub_pixel_variance4x8_sse4_1;
                if (flags & HAS_SSE4_1) aom_obmc_sub_pixel_variance64x128 = aom_obmc_sub_pixel_variance64x128_sse4_1;
                if (flags & HAS_SSE4_1) aom_obmc_sub_pixel_variance64x16 = aom_obmc_sub_pixel_variance64x16_sse4_1;
                if (flags & HAS_SSE4_1) aom_obmc_sub_pixel_variance64x32 = aom_obmc_sub_pixel_variance64x32_sse4_1;
                if (flags & HAS_SSE4_1) aom_obmc_sub_pixel_variance64x64 = aom_obmc_sub_pixel_variance64x64_sse4_1;
                if (flags & HAS_SSE4_1) aom_obmc_sub_pixel_variance8x16 = aom_obmc_sub_pixel_variance8x16_sse4_1;
                if (flags & HAS_SSE4_1) aom_obmc_sub_pixel_variance8x32 = aom_obmc_sub_pixel_variance8x32_sse4_1;
                if (flags & HAS_SSE4_1) aom_obmc_sub_pixel_variance8x4 = aom_obmc_sub_pixel_variance8x4_sse4_1;
                if (flags & HAS_SSE4_1) aom_obmc_sub_pixel_variance8x8 = aom_obmc_sub_pixel_variance8x8_sse4_1;
                if (flags & HAS_AVX2) aom_obmc_variance128x128 = aom_obmc_variance128x128_avx2;
                if (flags & HAS_AVX2) aom_obmc_variance128x64 = aom_obmc_variance128x64_avx2;
                if (flags & HAS_AVX2) aom_obmc_variance16x16 = aom_obmc_variance16x16_avx2;
                if (flags & HAS_AVX2) aom_obmc_variance16x32 = aom_obmc_variance16x32_avx2;
                if (flags & HAS_AVX2) aom_obmc_variance16x4 = aom_obmc_variance16x4_avx2;
                if (flags & HAS_AVX2) aom_obmc_variance16x64 = aom_obmc_variance16x64_avx2;
                if (flags & HAS_AVX2) aom_obmc_variance16x8 = aom_obmc_variance16x8_avx2;
                if (flags & HAS_AVX2) aom_obmc_variance32x16 = aom_obmc_variance32x16_avx2;
                if (flags & HAS_AVX2) aom_obmc_variance32x32 = aom_obmc_variance32x32_avx2;
                if (flags & HAS_AVX2) aom_obmc_variance32x64 = aom_obmc_variance32x64_avx2;
                if (flags & HAS_AVX2) aom_obmc_variance32x8 = aom_obmc_variance32x8_avx2;
                if (flags & HAS_AVX2) aom_obmc_variance4x16 = aom_obmc_variance4x16_avx2;
                if (flags & HAS_AVX2) aom_obmc_variance4x4 = aom_obmc_variance4x4_avx2;
                if (flags & HAS_AVX2) aom_obmc_variance4x8 = aom_obmc_variance4x8_avx2;
                if (flags & HAS_AVX2) aom_obmc_variance64x128 = aom_obmc_variance64x128_avx2;
                if (flags & HAS_AVX2) aom_obmc_variance64x16 = aom_obmc_variance64x16_avx2;
                if (flags & HAS_AVX2) aom_obmc_variance64x32 = aom_obmc_variance64x32_avx2;
                if (flags & HAS_AVX2) aom_obmc_variance64x64 = aom_obmc_variance64x64_avx2;
                if (flags & HAS_AVX2) aom_obmc_variance8x16 = aom_obmc_variance8x16_avx2;
                if (flags & HAS_AVX2) aom_obmc_variance8x32 = aom_obmc_variance8x32_avx2;
                if (flags & HAS_AVX2) aom_obmc_variance8x4 = aom_obmc_variance8x4_avx2;
                if (flags & HAS_AVX2) aom_obmc_variance8x8 = aom_obmc_variance8x8_avx2;
                if (flags & HAS_AVX2) eb_aom_variance4x4 = eb_aom_variance4x4_sse2;
                if (flags & HAS_AVX2) eb_aom_variance4x8 = eb_aom_variance4x8_sse2;
                if (flags & HAS_AVX2) eb_aom_variance4x16 = eb_aom_variance4x16_sse2;
                if (flags & HAS_AVX2) eb_aom_variance8x4 = eb_aom_variance8x4_sse2;
                if (flags & HAS_AVX2) eb_aom_variance8x8 = eb_aom_variance8x8_sse2;
                if (flags & HAS_AVX2) eb_aom_variance8x16 = eb_aom_variance8x16_sse2;
                if (flags & HAS_AVX2) eb_aom_variance8x32 = eb_aom_variance8x32_sse2;
                if (flags & HAS_AVX2) eb_aom_variance16x4 = eb_aom_variance16x4_avx2;
                if (flags & HAS_AVX2) eb_aom_variance16x8 = eb_aom_variance16x8_avx2;
                if (flags & HAS_AVX2) eb_aom_variance16x16 = eb_aom_variance16x16_avx2;
                if (flags & HAS_AVX2) eb_aom_variance16x32 = eb_aom_variance16x32_avx2;
                if (flags & HAS_AVX2) eb_aom_variance16x64 = eb_aom_variance16x64_avx2;
                if (flags & HAS_AVX2) eb_aom_variance32x8 = eb_aom_variance32x8_avx2;
                if (flags & HAS_AVX2) eb_aom_variance32x16 = eb_aom_variance32x16_avx2;
                if (flags & HAS_AVX2) eb_aom_variance32x32 = eb_aom_variance32x32_avx2;
                if (flags & HAS_AVX2) eb_aom_variance32x64 = eb_aom_variance32x64_avx2;
                if (flags & HAS_AVX2) eb_aom_variance64x16 = eb_aom_variance64x16_avx2;
                if (flags & HAS_AVX2) eb_aom_variance64x32 = eb_aom_variance64x32_avx2;
                if (flags & HAS_AVX2) eb_aom_variance64x64 = eb_aom_variance64x64_avx2;
                if (flags & HAS_AVX2) eb_aom_variance64x128 = eb_aom_variance64x128_avx2;
                if (flags & HAS_AVX2) eb_aom_variance128x64 = eb_aom_variance128x64_avx2;
                if (flags & HAS_AVX2) eb_aom_variance128x128 = eb_aom_variance128x128_avx2;
                if (flags & HAS_AVX2) eb_aom_highbd_10_variance8x8 = eb_aom_highbd_10_variance8x8_sse2;
                if (flags & HAS_AVX2) eb_aom_highbd_10_variance8x16 = eb_aom_highbd_10_variance8x16_sse2;
                if (flags & HAS_AVX2) eb_aom_highbd_10_variance8x32 = eb_aom_highbd_10_variance8x32_sse2;
                if (flags & HAS_AVX2) eb_aom_highbd_10_variance16x4 = eb_aom_highbd_10_variance16x4_avx2;
                if (flags & HAS_AVX2) eb_aom_highbd_10_variance16x8 = eb_aom_highbd_10_variance16x8_avx2;
                if (flags & HAS_AVX2) eb_aom_highbd_10_variance16x16 = eb_aom_highbd_10_variance16x16_avx2;
                if (flags & HAS_AVX2) eb_aom_highbd_10_variance16x32 = eb_aom_highbd_10_variance16x32_avx2;
                if (flags & HAS_AVX2) eb_aom_highbd_10_variance16x64 = eb_aom_highbd_10_variance16x64_avx2;
                if (flags & HAS_AVX2) eb_aom_highbd_10_variance32x8 = eb_aom_highbd_10_variance32x8_avx2;
                if (flags & HAS_AVX2) eb_aom_highbd_10_variance32x16 = eb_aom_highbd_10_variance32x16_avx2;
                if (flags & HAS_AVX2) eb_aom_highbd_10_variance32x32 = eb_aom_highbd_10_variance32x32_avx2;
                if (flags & HAS_AVX2) eb_aom_highbd_10_variance32x64 = eb_aom_highbd_10_variance32x64_avx2;
                if (flags & HAS_AVX2) eb_aom_highbd_10_variance64x16 = eb_aom_highbd_10_variance64x16_avx2;
                if (flags & HAS_AVX2) eb_aom_highbd_10_variance64x32 = eb_aom_highbd_10_variance64x32_avx2;
                if (flags & HAS_AVX2) eb_aom_highbd_10_variance64x64 = eb_aom_highbd_10_variance64x64_avx2;
                if (flags & HAS_AVX2) eb_aom_highbd_10_variance64x128 = eb_aom_highbd_10_variance64x128_avx2;
                if (flags & HAS_AVX2) eb_aom_highbd_10_variance128x64 = eb_aom_highbd_10_variance128x64_avx2;
                if (flags & HAS_AVX2) eb_aom_highbd_10_variance128x128 = eb_aom_highbd_10_variance128x128_avx2;
                if (flags & HAS_AVX2) eb_aom_quantize_b_64x64 = eb_aom_quantize_b_64x64_avx2;
                if (flags & HAS_AVX2) eb_aom_highbd_quantize_b_64x64 = eb_aom_highbd_quantize_b_64x64_avx2;
                if (flags & HAS_AVX2) eb_av1_fwd_txfm2d_16x8 = eb_av1_fwd_txfm2d_16x8_avx2;
                if (flags & HAS_AVX2) eb_av1_fwd_txfm2d_8x16 = eb_av1_fwd_txfm2d_8x16_avx2;
                if (flags & HAS_AVX2) eb_av1_fwd_txfm2d_16x4 = eb_av1_fwd_txfm2d_16x4_avx2;
                if (flags & HAS_AVX2) eb_av1_fwd_txfm2d_4x16 = eb_av1_fwd_txfm2d_4x16_avx2;
                if (flags & HAS_AVX2) eb_av1_fwd_txfm2d_8x4 = eb_av1_fwd_txfm2d_8x4_avx2;
                if (flags & HAS_AVX2) eb_av1_fwd_txfm2d_4x8 = eb_av1_fwd_txfm2d_4x8_avx2;
                if (flags & HAS_AVX2) eb_av1_fwd_txfm2d_32x8 = eb_av1_fwd_txfm2d_32x8_avx2;
                if (flags & HAS_AVX2) eb_av1_fwd_txfm2d_8x32 = eb_av1_fwd_txfm2d_8x32_avx2;
                if (flags & HAS_AVX2) eb_av1_fwd_txfm2d_64x64 = eb_av1_fwd_txfm2d_64x64_avx2;
                if (flags & HAS_AVX2) eb_av1_fwd_txfm2d_32x32 = eb_av1_fwd_txfm2d_32x32_avx2;
                if (flags & HAS_AVX2) eb_av1_fwd_txfm2d_16x16 = eb_av1_fwd_txfm2d_16x16_avx2;
                if (flags & HAS_AVX2) eb_av1_fwd_txfm2d_32x64 = eb_av1_fwd_txfm2d_32x64_avx2;
                if (flags & HAS_AVX2) eb_av1_fwd_txfm2d_64x32 = eb_av1_fwd_txfm2d_64x32_avx2;
                if (flags & HAS_AVX2) eb_av1_fwd_txfm2d_16x64 = eb_av1_fwd_txfm2d_16x64_avx2;
                if (flags & HAS_AVX2) eb_av1_fwd_txfm2d_64x16 = eb_av1_fwd_txfm2d_64x16_avx2;
                if (flags & HAS_AVX2) eb_av1_fwd_txfm2d_32x16 = eb_av1_fwd_txfm2d_32x16_avx2;
                if (flags & HAS_AVX2) eb_av1_fwd_txfm2d_16x32 = eb_av1_fwd_txfm2d_16x32_avx2;
#ifndef NON_AVX512_SUPPORT
                if (flags & HAS_AVX512F) {
                    eb_av1_fwd_txfm2d_64x64 = av1_fwd_txfm2d_64x64_avx512;
                    eb_av1_fwd_txfm2d_32x32 = av1_fwd_txfm2d_32x32_avx512;
                    eb_av1_fwd_txfm2d_16x16 = av1_fwd_txfm2d_16x16_avx512;
                    eb_av1_fwd_txfm2d_32x64 = av1_fwd_txfm2d_32x64_avx512;
                    eb_av1_fwd_txfm2d_64x32 = av1_fwd_txfm2d_64x32_avx512;
                    eb_av1_fwd_txfm2d_16x64 = av1_fwd_txfm2d_16x64_avx512;
                    eb_av1_fwd_txfm2d_64x16 = av1_fwd_txfm2d_64x16_avx512;
                    eb_av1_fwd_txfm2d_32x16 = av1_fwd_txfm2d_32x16_avx512;
                    eb_av1_fwd_txfm2d_16x32 = av1_fwd_txfm2d_16x32_avx512;
                }
#endif
                    if (flags & HAS_AVX2) eb_av1_fwd_txfm2d_8x8 = eb_av1_fwd_txfm2d_8x8_avx2;
                    if (flags & HAS_SSE4_1) eb_av1_fwd_txfm2d_4x4 = eb_av1_fwd_txfm2d_4x4_sse4_1;
                    if (flags & HAS_AVX2) handle_transform16x64 = handle_transform16x64_avx2;
                    if (flags & HAS_AVX2) handle_transform32x64 = handle_transform32x64_avx2;
                    if (flags & HAS_AVX2) handle_transform64x16 = handle_transform64x16_avx2;
                    if (flags & HAS_AVX2) handle_transform64x32 = handle_transform64x32_avx2;
                    if (flags & HAS_AVX2) handle_transform64x64 = handle_transform64x64_avx2;
                    if (flags & HAS_SSE2) eb_aom_fft4x4_float = eb_aom_fft4x4_float_sse2;
                    if (flags & HAS_AVX2) eb_aom_fft16x16_float = eb_aom_fft16x16_float_avx2;
                    if (flags & HAS_AVX2) eb_aom_fft32x32_float = eb_aom_fft32x32_float_avx2;
                    if (flags & HAS_AVX2) eb_aom_fft8x8_float = eb_aom_fft8x8_float_avx2;
                    if (flags & HAS_AVX2) eb_aom_ifft16x16_float = eb_aom_ifft16x16_float_avx2;
                    if (flags & HAS_AVX2) eb_aom_ifft32x32_float = eb_aom_ifft32x32_float_avx2;
                    if (flags & HAS_AVX2) eb_aom_ifft8x8_float = eb_aom_ifft8x8_float_avx2;
                    if (flags & HAS_SSE2) eb_aom_ifft4x4_float = eb_aom_ifft4x4_float_sse2;
                    if (flags & HAS_AVX2) av1_get_gradient_hist = av1_get_gradient_hist_avx2;
                    SET_AVX2_AVX512(
                        search_one_dual, search_one_dual_c, search_one_dual_avx2, search_one_dual_avx512);
                    SET_SSE41_AVX2(sad_loop_kernel_sparse,
                        sad_loop_kernel_sparse_c,
                        sad_loop_kernel_sparse_sse4_1_intrin,
                        sad_loop_kernel_sparse_avx2_intrin);
                    SET_SSE41_AVX2_AVX512(sad_loop_kernel,
                        sad_loop_kernel_c,
                        sad_loop_kernel_sse4_1_intrin,
                        sad_loop_kernel_avx2_intrin,
                        sad_loop_kernel_avx512_intrin);
                    SET_SSE41_AVX2(sad_loop_kernel_hme_l0,
                        sad_loop_kernel_c,
                        sad_loop_kernel_sse4_1_hme_l0_intrin,
                        sad_loop_kernel_avx2_hme_l0_intrin);

                    SET_SSE41(
                        svt_av1_apply_filtering, svt_av1_apply_filtering_c, svt_av1_apply_temporal_filter_sse4_1);
                    SET_AVX2(svt_av1_apply_temporal_filter_planewise,
                        svt_av1_apply_temporal_filter_planewise_c,
                        svt_av1_apply_temporal_filter_planewise_c);
                    SET_SSE41(svt_av1_apply_filtering_highbd,
                        svt_av1_apply_filtering_highbd_c,
                        svt_av1_highbd_apply_temporal_filter_sse4_1);
                    SET_AVX2_AVX512(combined_averaging_ssd,
                        combined_averaging_ssd_c,
                        combined_averaging_ssd_avx2,
                        combined_averaging_ssd_avx512);
                    SET_AVX2(ext_sad_calculation_8x8_16x16,
                        ext_sad_calculation_8x8_16x16_c,
                        ext_sad_calculation_8x8_16x16_avx2_intrin);
                    SET_SSE41(ext_sad_calculation_32x32_64x64,
                        ext_sad_calculation_32x32_64x64_c,
                        ext_sad_calculation_32x32_64x64_sse4_intrin);
                    SET_SSE2(sad_calculation_8x8_16x16,
                        sad_calculation_8x8_16x16_c,
                        sad_calculation_8x8_16x16_sse2_intrin);
                    SET_SSE2(sad_calculation_32x32_64x64,
                        sad_calculation_32x32_64x64_c,
                        sad_calculation_32x32_64x64_sse2_intrin);
                    SET_AVX2(ext_all_sad_calculation_8x8_16x16,
                        ext_all_sad_calculation_8x8_16x16_c,
                        ext_all_sad_calculation_8x8_16x16_avx2);
                    SET_AVX2(ext_eigth_sad_calculation_nsq,
                        ext_eigth_sad_calculation_nsq_c,
                        ext_eigth_sad_calculation_nsq_avx2);
                    SET_AVX2(ext_eight_sad_calculation_32x32_64x64,
                        ext_eight_sad_calculation_32x32_64x64_c,
                        ext_eight_sad_calculation_32x32_64x64_avx2);
                    SET_AVX2(eb_sad_kernel4x4, fast_loop_nxm_sad_kernel, eb_compute4x_m_sad_avx2_intrin);
                    SET_SSE41_AVX2_AVX512(get_eight_horizontal_search_point_results_8x8_16x16_pu,
                        get_eight_horizontal_search_point_results_8x8_16x16_pu_c,
                        get_eight_horizontal_search_point_results_8x8_16x16_pu_sse41_intrin,
                        get_eight_horizontal_search_point_results_8x8_16x16_pu_avx2_intrin,
                        get_eight_horizontal_search_point_results_8x8_16x16_pu_avx512_intrin);
                    SET_SSE41_AVX2(get_eight_horizontal_search_point_results_32x32_64x64_pu,
                        get_eight_horizontal_search_point_results_32x32_64x64_pu_c,
                        get_eight_horizontal_search_point_results_32x32_64x64_pu_sse41_intrin,
                        get_eight_horizontal_search_point_results_32x32_64x64_pu_avx2_intrin);
                    SET_SSE2(
                        initialize_buffer_32bits, initialize_buffer_32bits_c, initialize_buffer_32bits_sse2_intrin);
                    SET_AVX2(nxm_sad_kernel_sub_sampled,
                        nxm_sad_kernel_helper_c,
                        nxm_sad_kernel_sub_sampled_helper_avx2);

                    SET_AVX2(nxm_sad_kernel, nxm_sad_kernel_helper_c, nxm_sad_kernel_helper_avx2);
                    SET_AVX2(nxm_sad_avg_kernel, nxm_sad_avg_kernel_helper_c, nxm_sad_avg_kernel_helper_avx2);
                    SET_SSE2_AVX2(
                        compute_mean_8x8, compute_mean_c, compute_mean8x8_sse2_intrin, compute_mean8x8_avx2_intrin);
                    SET_SSE2(compute_mean_square_values_8x8,
                        compute_mean_squared_values_c,
                        compute_mean_of_squared_values8x8_sse2_intrin);
                    SET_SSE2_AVX2(compute_interm_var_four8x8,
                        compute_interm_var_four8x8_c,
                        compute_interm_var_four8x8_helper_sse2,
                        compute_interm_var_four8x8_avx2_intrin);
                    SET_AVX2(sad_16b_kernel, sad_16b_kernel_c, sad_16bit_kernel_avx2);
                    SET_AVX2(av1_compute_cross_correlation,
                        av1_compute_cross_correlation_c,
                        av1_compute_cross_correlation_avx2);
                    SET_AVX2(av1_k_means_dim1, av1_k_means_dim1_c, av1_k_means_dim1_avx2);
                    SET_AVX2(av1_k_means_dim2, av1_k_means_dim2_c, av1_k_means_dim2_avx2);
                    SET_AVX2(av1_calc_indices_dim1, av1_calc_indices_dim1_c, av1_calc_indices_dim1_avx2);
                    SET_AVX2(av1_calc_indices_dim2, av1_calc_indices_dim2_c, av1_calc_indices_dim2_avx2);
                    if (flags & HAS_SSE2) eb_av1_get_nz_map_contexts = eb_av1_get_nz_map_contexts_sse2;

#if RESTRUCTURE_SAD
                    SET_AVX2(pme_sad_loop_kernel, pme_sad_loop_kernel_c, pme_sad_loop_kernel_avx2);
#endif

#endif

}
