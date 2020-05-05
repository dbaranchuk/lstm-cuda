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
	cout << "Program initializing" << endl;
	if (argc < 4) {
		cout << argv[0] << " <learning rate> <blocks> <cells>" << endl;
		return -1;
	}
	std::srand(unsigned(std::time(0)));

    long long networkStart, networkEnd;
    networkStart = getMSec();
    DatasetAdapter dataset = Dataset();
    networkEnd = getMSec();
    cout << "Language Dataset loaded in " << (networkEnd - networkStart) << "msecs" << endl;

	int num_epochs = 10;
	int num_batches = 256;
	int emb_size = 128;
	int num_classes = 10;
	int blocks = atoi(argv[2]);
	int cells = atoi(argv[3]);
	int seq_len = 20;

	double mse = 0;
	double learningRate = atof(argv[1]);

	TextClassifier model = TextClassifier(emb_size, blocks, cells,
	                                     learningRate, num_classes);
	Data data = Data(emb_size, num_classes);
	cout << "Network initialized" << endl;

	for (int e = 0; e < num_epochs; e++) {
		networkStart = getMSec();
		double loss = 0.0;
		for (int i = 0; i < num_batches; i++) {
			vector<double> onehot_target = get_onehot_target(std::rand() % num_classes);
			vector<vector<double>> embs = get_emb_sequence(seq_len);
            loss += model.train(embs, onehot_target);
		}
		loss /= num_batches;
		networkEnd = getMSec();

		cout << "Epoch " << e << " completed in " << (networkEnd - networkStart) << "msecs" << endl;
		cout << "Loss[" << e << "] = " << loss << endl;
		//cout << "Accuracy[" << e << "] = " << (100.0 * (float)c / (float)n) << endl;
		dataset.reset();
	}
	return 0;
}
