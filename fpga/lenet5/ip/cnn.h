#include <hls_stream.h>
#include <ap_axi_sdata.h>

#define bitwidth 32
#define InpWidth 8
#define quant_scale 128// 2^(wordwidth-1)

#define INF 999999

#ifndef AXI_VAL_DEF
typedef ap_int<18> CNN_VAL;
//typedef short CNN_VAL;
typedef int SUM_VAL;
struct AXI_DMA_IO{
	ap_int<bitwidth> data;
	bool last;
};
#define AXI_VAL_DEF
#endif

void cnn(hls::stream<AXI_DMA_IO> &in_stream,hls::stream<AXI_DMA_IO> &out_stream);
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) > (y)) ? (y) : (x))
const CNN_VAL minval = -255;
const CNN_VAL maxval = 255;
