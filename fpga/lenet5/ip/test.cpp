#include "cnn.h"
#include <stdio.h>

#include <vector>
#include <string>
#include <fstream>
#include<iostream>

using namespace std;
int main(int argc, char *argv[])
{
	std::ifstream infile;
	infile.open("E:\\currentwork\\fpga\\hls\\src\\lenet5\\cnnip\\cnnip\\weight.txt", ios::in);
	hls::stream<AXI_DMA_IO> in;
	hls::stream<AXI_DMA_IO> out;
	AXI_DMA_IO i;
	AXI_DMA_IO o;
	i.data = 101;
	i.last = 0;
	in.write(i);
	int data;

	//�����������
    if(!infile.is_open ())
        cout << "Open file failure" << endl;
	while(!infile.eof()) {
		infile >> data;
		i.data = data;
		i.last = 0;
		in.write(i);
	}
	infile.close();
	cnn(in,out);
	//����ͼƬ
	infile.open("E:\\currentwork\\fpga\\hls\\src\\lenet5\\cnnip\\cnnip\\image.txt", ios::in);
	i.data = 301;
	i.last = 0;
	in.write(i);
	i.data = 1;
	i.last = 0;
	in.write(i);
	//�����������
    if(!infile.is_open ())
        cout << "Open file failure" << endl;
	while(!infile.eof()) {
		infile >> data;
		i.data = data;
		i.last = 0;
		in.write(i);
	}
	infile.close();
	cnn(in,out);
	for(int i=0;i<1;i++) {
		o = out.read();
		cout<< o.data << " ";
	}

}
