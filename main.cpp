#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

vector<string> loadLabels(const string& filename) {
    vector<string> classes;
    ifstream ifs(filename.c_str());
    string line;
    while (getline(ifs, line)) {
        classes.push_back(line);
    }
    return classes;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: ./image_classifier <image_path>" << endl;
        return -1;
    }

    string imagePath = argv[1];
    Mat image = imread(imagePath);
    if (image.empty()) {
        cerr << "Could not read input image!" << endl;
        return -1;
    }

    // Load the pre-trained model
    Net net = readNetFromONNX("mobilenetv2.onnx");

    // Preprocessing: resize and normalize
    Mat blob = blobFromImage(image, 1.0 / 255.0, Size(224, 224), Scalar(0, 0, 0), true, false);

    net.setInput(blob);
    Mat prob = net.forward();

    Point classIdPoint;
    double confidence;

    minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
    int classId = classIdPoint.x;

    vector<string> labels = loadLabels("labels.txt");

    cout << "Predicted class: " << labels[classId] << " (" << confidence * 100 << "% confidence)" << endl;

    return 0;
}
