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
		cout << argv[0] << " <learning rate> <blocks> <cells> <size ...>" << endl;
		return -1;
	}

	int updatePoints = 10;
	int maxEpoch = 10;
	int trainingSize = 500;
	int blocks = atoi(argv[2]);
	int cells = atoi(argv[3]);
	int sumNeurons = (blocks * cells);
	double mse = 0;
	double learningRate = atof(argv[1]);
	long long networkStart, networkEnd, sumTime = 0;

	const int _day = getDate()->tm_mday;

	networkStart = getMSec();
	DatasetAdapter dataset = DatasetAdapter();
	networkEnd = getMSec();
	cout << "Language Dataset loaded in " << (networkEnd - networkStart) << "msecs" << endl;


	LSTMNetwork network = LSTMNetwork(dataset.getCharSize(), blocks, cells, learningRate);
	OutputTarget target = OutputTarget(dataset.getCharSize(), dataset.getCharSize());
	cout << "Network initialized" << endl;


	for (int i = 0; i < (argc - 5); i++) {
		network.addLayer(atoi(argv[5 + i]));
		sumNeurons += atoi(argv[5 + i]);
	} network.addLayer(dataset.getCharSize());


	int totalIterations = 0;
	for (int e = 0; e < maxEpoch; e++) {
		int c = 0, n = 0;
		vector<double> error, output;

		networkStart = getMSec();
		for (int i = 0; i < trainingSize && dataset.nextChar(); i++) {
			DatasetExample data = dataset.getChar();
			error = network.train(target.getOutputFromTarget(data.current),
					target.getOutputFromTarget(data.next));
		}

		dataset.reset();

		for (int i = 0; i < trainingSize && dataset.nextChar(); i++) {
			DatasetExample data = dataset.getChar();
			output = network.classify(target.getOutputFromTarget(data.current));

			n++;
			if (target.getTargetFromOutput(output) == (int)data.next) c++;
		} networkEnd = getMSec();

		sumTime += (networkEnd - networkStart);
		totalIterations += 1;

		mse = 0;
		for (int i = 0; i < error.size(); i++)
			mse += error[i] * error[i];
		mse /= error.size() * 2;

		if (((e + 1) % (maxEpoch / updatePoints)) == 0) {
			cout << "Epoch " << e << " completed in " << (networkEnd - networkStart) << "msecs" << endl;
			cout << "Error[" << e << "] = " << mse << endl;
			cout << "Accuracy[" << e << "] = " << (100.0 * (float)c / (float)n) << endl;
		}

		dataset.reset();
	}

	//vector<vector<double> > seed;
	//seed.push_back(target.getOutputFromTarget((int)'I'));
	//for (int i = 0; i < 500; i++) {
	//	vector<double> output = network.classify(seed[i]);
	//	seed.push_back(output);
    //	char text = (char)target.getTargetFromOutput(output);
	//}
	return 0;
}
