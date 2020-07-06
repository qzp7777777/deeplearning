#include "cnn.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) > (y)) ? (y) : (x))

static CNN_VAL conv1[5][5][6];

static CNN_VAL conv1_b[6];
static CNN_VAL conv2[5][5][6][16];

static CNN_VAL conv2_b[16];
static CNN_VAL fn1[256][120];

static CNN_VAL fn1_b[120];
static CNN_VAL fn2[120][84];

static CNN_VAL fn2_b[84];
static CNN_VAL fn3[84][10];
static CNN_VAL fn3_b[10];
static short fresult[10000];
template<int x> void readbias(CNN_VAL bias[x],hls::stream<AXI_DMA_IO> &in_stream) {
	AXI_DMA_IO in;
	for(int i=0;i < x; i ++) {
		in = in_stream.read();
		bias[i] = in.data;
	}
}

template<int x,int y> void readfn(CNN_VAL matrix[x][y],hls::stream<AXI_DMA_IO> &in_stream) {
	AXI_DMA_IO in;
	for(int i=0;i < x; i ++) {
		for(int j=0;j < y; j ++) {
			in = in_stream.read();
			matrix[i][j] = in.data;
		}
	}
}

template<int x,int y,int z> void pool(CNN_VAL in[x][y][z],CNN_VAL out[x>>1][y>>1][z]) {
	pool:
	for(int i=0;i < x; i = i + 2) {
		pool1_1:
		for(int j=0;j < y; j =j + 2) {
			pool1_2:
			for(int k=0;k < z; k ++) {
#pragma HLS PIPELINE II=1
				CNN_VAL val1 = MAX(in[i][j][k],in[i][j+1][k]);
				CNN_VAL val2 = MAX(in[i+1][j][k],in[i+1][j+1][k]);
				CNN_VAL val3 = MAX(val1,val2);
				val3 = MAX(val3,minval);
				val3 = MIN(val3,maxval);
				out[i>>1][j>>1][k] = val3;
			}
		}
	}
}
void load(CNN_VAL img[28][28],hls::stream<AXI_DMA_IO> &in_stream){
	AXI_DMA_IO in;
	load:
	for(int i=0;i < 28; i ++) {
		load_1:
		for(int j=0;j < 28; j ++) {
#pragma HLS pipeline
			in = in_stream.read();
			img[i][j] = in.data;
		}
	}
}

void cov1(CNN_VAL img[28][28],CNN_VAL conv1Result[24][24][6]) {
	cov1:
		cov1_1:
	for(int i=0;i < 28 - 4; i ++) {
		cov1_2:
		for(int j=0;j < 28 - 4; j ++) {
			for(int k=0;k < 6; k ++) {
#pragma HLS PIPELINE II=1 rewind
				SUM_VAL sum = 0;
				for(int x=0;x < 5; x ++) {
					for(int y=0;y < 5; y ++) {
						sum = sum + img[i+x][j+y]*conv1[x][y][k];
					}
				}
				conv1Result[i][j][k]= (sum + conv1_b[k])>>8;
			}
		}
	}
}

void cov2(CNN_VAL pool1Result[12][12][6],CNN_VAL conv2Result[8][8][16]) {
	cov2:
	for(int i=0;i < 12 - 4; i ++) {
		cov2_1:
		for(int j=0;j < 12 - 4; j ++) {
			for(int k=0;k < 16; k ++) {
#pragma HLS PIPELINE II=1 rewind
				SUM_VAL sum = 0;
				for(int x=0;x < 5; x ++) {
					for(int y=0;y < 5; y ++) {
						for(int z=0;z < 6; z ++) {
							sum = sum + pool1Result[i+x][j+y][z]*conv2[x][y][z][k];
						}
					}
				}
				conv2Result[i][j][k]= (sum + conv2_b[k])>>8;
			}
		}
	}
}
void copy(CNN_VAL pool2Result[4][4][16],hls::stream<CNN_VAL> &p2_fn1){
	cpy:for(int i=0;i < 4; i ++) {
		for(int k=0;k < 4; k ++) {
			for(int p=0;p < 16; p ++) {
#pragma HLS pipeline
				p2_fn1.write(pool2Result[i][k][p]);
			}
		}
	}
}
void fn_1(hls::stream<CNN_VAL> &p2_fn1,CNN_VAL fn1Result[120]) {
	CNN_VAL fnbegin[4*4*16];
#pragma HLS ARRAY_PARTITION variable=fnbegin cyclic factor=16
	for(int i=0;i < 256; i ++) {
		fnbegin[i] = p2_fn1.read();
	}
	fn1:
		for(int j=0;j < 120; j++) {
			SUM_VAL sum=0;
			fn1_1:
			for(int i=0;i < 256; i ++) {
#pragma HLS PIPELINE
#pragma HLS UNROLL factor=16
				sum = sum + (SUM_VAL)fn1[i][j] * fnbegin[i];
			}
			sum = (sum + fn1_b[j])>>8;
			CNN_VAL fr = sum;
			fr = MAX(fr,minval);
			fr = MIN(fr,maxval);
			fn1Result[j] = fr;
	}
}
void fn_2(CNN_VAL fn1Result[120],CNN_VAL fn2Result[84]) {
	fn2:
	for(int j=0;j < 84; j++) {

		SUM_VAL sum = 0;
		for(int i=0;i < 120; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=10
			sum = sum + fn2[i][j] * fn1Result[i];
		}
		CNN_VAL fr = (sum + fn2_b[j])>>8;
		fr = MAX(fr,minval);
		fr = MIN(fr,maxval);
		fn2Result[j] = fr;
	}
}
void fn_3(CNN_VAL fn2Result[84],SUM_VAL fn3Result[10]) {
	fn3:
	for(int j=0;j < 10; j++) {
		SUM_VAL sum = 0;
		for(int i=0;i < 84; i++) {
	#pragma HLS PIPELINE II=1
			sum = sum + fn3[i][j] * fn2Result[i];
		}
		sum = sum + fn3_b[j];
		fn3Result[j] = sum;
	}
}
void writeout(SUM_VAL fn3Result[10],int n){
	AXI_DMA_IO out;
	writeout:
	int result = 0;
	int max = -1;
	calresut:for(int j=0;j < 10; j++) {
#pragma HLS PIPELINE II=1
		out.data = fn3Result[j];
		if(fn3Result[j] > max) {
			max = fn3Result[j];
			result = j;
		}
	}
	fresult[n] = result;
}
void cnn(hls::stream<AXI_DMA_IO> &in_stream,hls::stream<AXI_DMA_IO> &out_stream) {

#pragma HLS INTERFACE axis port=in_stream
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS ARRAY_PARTITION variable=conv1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=conv1 complete dim=2
//#pragma HLS ARRAY_PARTITION variable=conv2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=conv2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=conv2 complete dim=3
#pragma HLS ARRAY_PARTITION variable=fn1 cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=fn2 cyclic factor=10 dim=1
	static int num=0;
	AXI_DMA_IO in;
	AXI_DMA_IO out;

	if(num>0){
	}
	//传过来的数据类型
	CNN_VAL type;
	in = in_stream.read();
	type = in.data;
	//type为0 认为传过的是卷积参数
	if(type==101) {

		for(int i=0;i < 5; i ++) {
			for(int j=0;j < 5; j ++) {
				for(int k=0;k < 6; k ++) {
					in = in_stream.read();
					conv1[i][j][k] = in.data;
				}
			}
		}
		readbias<6>(conv1_b,in_stream);
		for(int i=0;i < 5; i ++) {
			for(int j=0;j < 5; j ++) {
				for(int k=0;k < 6; k ++) {
					for(int f=0;f < 16; f ++) {
						in = in_stream.read();
						conv2[i][j][k][f] = in.data;
					}
				}
			}
		}
		readbias<16>(conv2_b,in_stream);
		readfn<256,120>(fn1,in_stream);
		readbias<120>(fn1_b,in_stream);
		readfn<120,84>(fn2,in_stream);
		readbias<84>(fn2_b,in_stream);
		readfn<84,10>(fn3,in_stream);
		readbias<10>(fn3_b,in_stream);

	}
	if(type==301) {
		in = in_stream.read();
		num = in.data;
		for(int n=0;n<num;n++) {
			CNN_VAL img[28][28];
#pragma HLS ARRAY_PARTITION variable=img cyclic factor=5 dim=1
#pragma HLS ARRAY_PARTITION variable=img cyclic factor=5 dim=2
			CNN_VAL conv1Result[24][24][6];
#pragma HLS ARRAY_PARTITION variable=conv1Result cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=conv1Result cyclic factor=2 dim=2
			CNN_VAL pool1Result[12][12][6];
//#pragma HLS ARRAY_PARTITION variable=pool1Result cyclic factor=5 dim=1
#pragma HLS ARRAY_PARTITION variable=pool1Result cyclic factor=5 dim=2
#pragma HLS ARRAY_PARTITION variable=pool1Result complete dim=3
			CNN_VAL conv2Result[8][8][16];
			CNN_VAL pool2Result[4][4][16];

			hls::stream<CNN_VAL> p2_fn1;
#pragma HLS STREAM variable=p2_fn1 depth=512 dim=1
			CNN_VAL fn1Result[120];
#pragma HLS ARRAY_PARTITION variable=fn1Result cyclic factor=10
			CNN_VAL fn2Result[84];
		//#pragma HLS ARRAY_PARTITION variable=fn2Result complete dim=1
			SUM_VAL fn3Result[10];

			load(img,in_stream);
			cov1(img,conv1Result);
			pool<24,24,6>(conv1Result,pool1Result);
			cov2(pool1Result,conv2Result);
			pool<8,8,16>(conv2Result,pool2Result);
			copy(pool2Result,p2_fn1);
			fn_1(p2_fn1,fn1Result);
			fn_2(fn1Result,fn2Result);
			fn_3(fn2Result,fn3Result);
			writeout(fn3Result,n);
		}
		for(int n=0;n<num;n++){
			if(num-1==n) {
				out.last = 1;
			}else{
				out.last = 0;
			}
			out.data = fresult[n];
			out_stream.write(out);
		}

	}

}
