#include "Data.h"

long long Data::n = 0;

Data::Data(int _emb_size, int _num_classes) {
	emb_size = _emb_size;
	num_classes = _num_classes;

	for (int i = 0; i < num_classes; i++) {
		vector<double> tmp;
		for (int j = 0; j < num_classes; j++) {
			if (i == j) tmp.push_back(1.0);
			else tmp.push_back(0.0);
		}
		onehot_targets.push_back(tmp);
	}
}


Data::~Data() {}


vector<double> Data::get_onehot_target(int class_id) {
	return onehot_targets[class_id];
}


vector<double> Data::get_emb_sequence(int seq_len) {
	default_random_engine g(time(0) + (n++));
	normal_distribution<double> d(0, 1);
	vector<double> sequence(seq_len * emb_size);
	for (int i = 0; i < seq_len; i++) {
		for (int j = 0; j < emb_size; j++)
			sequence[i*emb_size + j] = d(g);
	}
	return sequence;
}

//int OutputTarget::getTargetFromOutput(vector<double> output) {
//	for (int i = 0; i < classes; i++) {
//		bool matches = true;
//		for (int j = 0; j < nodes; j++) {
//			if (abs(output[j] - classifiers[i][j]) >= 1) {
//				matches = false;
//				break;
//			}
//		}
//		if (matches) return i;
//	}
//	return -1;
//}

