#include "TextClassifier.cuh"
#include "Data.h"
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>

using namespace std;

long long getMSec() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

struct tm *getDate() {
	time_t t = time(NULL);
	struct tm *timeObject = localtime(&t);
	return timeObject;
}

int main(int argc, char *argv[]) {
	std::srand(unsigned(std::time(0)));

    long long networkStart, networkEnd;
    networkStart = getMSec();

	int num_epochs = 10;
	int num_batches = 100;
	int emb_size = 128;
	int num_classes = 10;
	int cells = 100;//atoi(argv[2]);
	int seq_len = 100;

	double learningRate = 0.01;//atof(argv[1]);

	TextClassifier model = TextClassifier(emb_size, cells, learningRate, num_classes);
	Data data = Data(emb_size, num_classes);
	networkEnd = getMSec();
	cout << "Network initialized in " << (networkEnd - networkStart) << "msecs" << endl;

	for (int e = 0; e < num_epochs; e++) {
		networkStart = getMSec();
		double loss = 0.0;
		for (int i = 0; i < num_batches; i++) {
			vector<double> onehot_target = data.get_onehot_target(std::rand() % num_classes);
			vector<double> embs = data.get_emb_sequence(seq_len);
            loss += model.train(embs, onehot_target);
		}
		loss /= num_batches;
		networkEnd = getMSec();

		cout << "Epoch " << e << " completed in " << (networkEnd - networkStart) << "msecs" << endl;
		cout << "Loss[" << e << "] = " << loss << endl;
		//cout << "Accuracy[" << e << "] = " << (100.0 * (float)c / (float)n) << endl;
	}
	return 0;
}
