#include "LSTMNetwork.cuh"
#include "DatasetAdapter.h"
#include "OutputTarget.h"
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
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

    long long networkStart, networkEnd, sumTime = 0;
    networkStart = getMSec();
    DatasetAdapter dataset = DatasetAdapter();
    networkEnd = getMSec();
    cout << "Language Dataset loaded in " << (networkEnd - networkStart) << "msecs" << endl;

	int maxEpoch = 10;
	int trainingSize = 256;
	int emb_size = 64;
	int num_classes = dataset.getCharSize();
	int blocks = atoi(argv[2]);
	int cells = atoi(argv[3]);
	int seq_len = 10;

	double mse = 0;
	double learningRate = atof(argv[1]);

	LSTMNetwork network = LSTMNetwork(emb_size, blocks, cells,
	                                  learningRate, num_classes);
	OutputTarget target = OutputTarget(emb_size, num_classes);
	cout << "Network initialized" << endl;

	for (int e = 0; e < maxEpoch; e++) {
		//int c = 0, n = 0;
		vector<double> error, output;

		networkStart = getMSec();
		for (int i = 0; i < trainingSize && dataset.nextChar(); i++) {
			DatasetExample data = dataset.getChar();
			vector<vector<double>> inputs;
            vector<vector<double>> targets;
			for (int j=0; j < seq_len; j++) {
                inputs.push_back(target.getOutputFromTarget(data.current));
                targets.push_back(target.getOutputFromTarget(data.next));
            }
            error = network.train(inputs, targets);
		}

		dataset.reset();

		//for (int i = 0; i < trainingSize && dataset.nextChar(); i++) {
		//	DatasetExample data = dataset.getChar();
		//	output = network.classify(target.getOutputFromTarget(data.current));

		//	n++;
		//	if (target.getTargetFromOutput(output) == (int)data.next) c++;
		//}
		networkEnd = getMSec();
		sumTime += (networkEnd - networkStart);

		mse = 0;
		for (int i = 0; i < error.size(); i++)
			mse += error[i] * error[i];
		mse /= error.size() * 2;

		cout << "Epoch " << e << " completed in " << (networkEnd - networkStart) << "msecs" << endl;
		cout << "Error[" << e << "] = " << mse << endl;
		//cout << "Accuracy[" << e << "] = " << (100.0 * (float)c / (float)n) << endl;
		dataset.reset();
	}
	return 0;
}
