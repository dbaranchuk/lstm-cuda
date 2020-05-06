#ifndef DATA_H_
#define DATA_H_

#include <vector>
#include <cmath>
#include <iostream>
#include <random>

using namespace std;

class Data {
private:
	static long long n;
	int emb_size = 64;
	int num_classes = 10;
	vector<vector<double> > onehot_targets;
public:
	Data(int emb_size, int num_classes);
	~Data();

	vector<double> get_onehot_target(int class_idx);
	vector<double> get_emb_sequence(int seq_len);
	//int getTargetFromOutput(vector<double> output);
};

#endif /* DATA_H_ */
