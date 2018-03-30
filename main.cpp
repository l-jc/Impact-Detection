#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iterator>
#include <float.h>
#include <assert.h>
#include <cstdlib>
using namespace std;

double dtwDistance(vector<double> s, vector<double> t);
bool detectPeak(double ac, double &posVal, double &negVal);
void analyzeData(vector<double> ax, vector<double> ay, vector<double> rz, vector<vector<double> > &event);
double distanceRatio(vector<double> event, vector<vector<double> > templates);
vector<double> readData(string filename, int ncol, bool restore);
void kMeans(const vector<vector<double> > &templates, int k, vector<vector<double> > &centroids, vector<int> &belonging);
vector<vector<double> > topKSim(const vector<vector<double> > &templates, int k);
vector<string> readFileList(string fname);

double posThresh = 1.0;
double negThresh = -1.0;
int createdFrameLength = 500;

vector<double> readData(string filename, int ncol, bool restore=false)
{
    double g = 9.8; //gravity
    vector<double> signal;
    ifstream file;
    file.open(filename);
    if (!file) {
        cout << "file not found: " << filename << endl;
        exit(-1);
    }
    while (!file.eof()) {
        double tmp = 0;
        for (int i=0;i<ncol-1;++i) file >> tmp;
        file >> tmp;
        if (restore) tmp *= g;
        signal.push_back(tmp);
    }
    return signal;
}

double dtwDistance(vector<double> s, vector<double> t)
{
    int sLength = s.size();
    int tLength = t.size();
    double **dtw = new double*[sLength+1];
    for (int i=0;i<=sLength;++i) dtw[i] = new double[tLength+1];
    for (int i=0;i<=sLength;++i) dtw[i][0] = DBL_MAX;
    for (int i=0;i<=tLength;++i) dtw[0][i] = DBL_MAX;

    dtw[0][0] = 0;
    for (int i=1;i<=sLength;++i)
    for (int j=1;j<=tLength;++j) {
        double cost = fabs(s[i] - t[j]);
        dtw[i][j] = cost + min(dtw[i-1][j],min(dtw[i][j-1],dtw[i-1][j-1]));
    }

    return dtw[sLength][tLength];
}

bool detectPeak(double ac, double &posVal, double &negVal)
{
    bool isPeak = false;
    if (ac >= posThresh) {
        if (ac >= posVal) posVal = ac;
        else isPeak = true;
    }
    else if (ac < posThresh && posVal > 0) isPeak = true;

    if (ac <= negThresh) {
        if (ac <= negVal) negVal = ac;
        else isPeak = true;
    }
    else if (ac > negThresh && negVal < 0) isPeak = true;

    return isPeak;
}

void analyzeData(vector<double> ax, vector<double> ay, vector<double> rz, vector<vector<double> > &event)
{
    double lastYPosVal(0), lastYNegVal(0), lastXPosVal(0), lastXNegVal(0);
    assert(ay.size()==ax.size());
    int n = ay.size();
    for (int i=0;i<n;++i) {
        bool isYPeak = detectPeak(ay[i],lastYPosVal,lastYNegVal);
        if (isYPeak) {
            bool isXPeak = detectPeak(ax[i],lastXPosVal,lastXNegVal);
            if (isYPeak && isXPeak) {
                // createEventFrame(event);
                vector<double> x(createdFrameLength), y(createdFrameLength), z(createdFrameLength);
                int e;
                if (i+createdFrameLength <= ax.size()) e = i + createdFrameLength; else e = ax.size();
                copy(ax.begin()+i,ax.begin()+i+createdFrameLength,x.begin());
                event.push_back(x);
                if (i+createdFrameLength <= ay.size()) e = i + createdFrameLength; else e = ay.size();
                copy(ay.begin()+i,ay.begin()+i+createdFrameLength,y.begin());
                event.push_back(y);
                if (i+createdFrameLength <= rz.size()) e = i + createdFrameLength; else e = rz.size();
                copy(rz.begin()+i,rz.begin()+i+createdFrameLength,z.begin());
                event.push_back(z);
                i = i + createdFrameLength;
            }
        }
    }
}

double distanceRatio(vector<double> event, vector<vector<double> > templates)
{
    int numTemplates = templates.size();
    double dmax = -DBL_MAX;
    double dmin = DBL_MAX;
    for (int i=0;i<numTemplates;++i) {
        vector<double> t = templates[i];
        double sim = dtwDistance(event,t);
        if (sim < dmin) {
            dmin = sim;
        }
        else if (sim > dmax) {
            dmax = sim;
        }
    }
    return dmin / dmax;
}

void kMeans(const vector<vector<double> > &templates, int k, vector<vector<double> > &centroids, vector<int> &belonging)
{
    assert(centroids.size()==k);
    assert(belonging.size()==templates.size());
    int nTemplates = templates.size();
    int nSize = templates[0].size();

//    for (int i=0;i<k;++i)
//        centroids.push_back(templates[i]);
//    for (int i=0;i<nTemplates;++i)
//        belonging[i] = -1;

    bool change = true;
    int numIter = 0;
    while (change) {
        numIter++;
        change = false;
        for (int i=0;i<nTemplates;++i) {
            double mindist = DBL_MAX;
            int c = -1;
            for (int j=0;j<k;++j) {
                double dist = dtwDistance(templates[i],centroids[j]);
                if (dist < mindist) {
                    mindist = dist;
                    c = j;
                }
            }
            if (c != belonging[i] && c != -1) {
                change = true;
                belonging[i] = c;
            }
        }
        // update centroids
        for (int i=0;i<k;++i) {
            int n = 0;
            vector<double> sum(nSize,0);
            for (int j=0;j<nTemplates;++j) {
                if (belonging[j] == i) {
                    for (int l=0;l<nSize;++l) {
                        sum[l] += templates[j][l];
                    }
                    n++;
                }
            }
            for (int l=0;l<nSize;++l) {
                sum[l] = sum[l] / n;
            }
            centroids[i] = sum;
        }
    }
//    cout << "number of iterations: " << numIter << endl;
    return ;
}

int findSmallestCluster(const vector<vector<double> > &data, const vector<vector<double> > &centroids, const vector<int> &belonging)
{
    double minvalue = DBL_MAX;
    int c = -1;
    for (int i=0;i<centroids.size();++i) {
        double value = 0;
        int n = 0;
        vector<double> center = centroids[i];
        for (int j=0;j<data.size();++j) {
            if (belonging[j] == i) {
                value += dtwDistance(center,data[j]);
                n++;
            }
        }
        value = value / n;
        if (value < minvalue) {
            minvalue = value;
            c = n;
        }
    }
    return c;
}

void pruning(const vector<vector<double> > &data, int k, vector<vector<double> > &prunedData)
{
    vector<vector<double> > centroids;
    for (int i=0;i<k;++i) centroids.push_back(data[i]);
    vector<int> belonging(data.size(),-1);
    kMeans(data,k,centroids,belonging);
    int c = findSmallestCluster(data,centroids,belonging);
    for (int i=0;i<data.size();++i) {
        if (belonging[i] == c) {
            prunedData.push_back(data[i]);
        }
    }
}

vector<vector<double> > topKSim(const vector<vector<double> > &templates, int k)
{
    int nTemplates = templates.size();
    vector<bool> belonging(nTemplates,false);
    for (int i=0;i<k;++i)
        belonging[i] = true;

    bool change = true;
    int numIter = 0;
    while (change) {
        change = false;
        numIter++;
        for (int i=0;i<nTemplates;++i) {
            if (belonging[i] == false) {
                //计算templates[i]和topk的距离和topk[j]和topk的距离。近的那个加入topk。
                // compare it with topk[j]
                for (int j=0;j<nTemplates;++j) {
                    if (belonging[j] == true) {
                        double indist = 0;
                        double outdist = 0;
                        for (int l=0;l<nTemplates;++l) {
                            if (belonging[l] == true && l!=j) {
                                indist += dtwDistance(templates[j],templates[l]);
                                outdist += dtwDistance(templates[i],templates[l]);
                            }
                        }
                        if (outdist < indist) {
                            belonging[i] = true;
                            belonging[j] = false;
                            change = true;
                            break;
                        }
                    }
                }
            }
        }
    }
    vector<vector<double> > topk;
    for (int i=0;i<nTemplates;++i)
        if (belonging[i] == true) {
            topk.push_back(templates[i]);
//            cout << i << " ";
        }
//    cout << endl;
//    cout << "number of iterations: " << numIter << endl;
    return topk;
}

void createTemplate(vector<string> filenames, vector<vector<vector<double> > > &templates)
{
    /** read input files **/
    int numFiles = filenames.size();
    vector<vector<double> > coarse;
    for (int i=0;i<numFiles;i=i+3) {
        coarse.push_back(readData(filenames[i],2,false));
        coarse.push_back(readData(filenames[i+1],2,false));
        coarse.push_back(readData(filenames[i+2],2,false));
    }
    /********************/

    /** butterworth low pass filter **/
    /**      not implemented        **/

    /** detect events **/
    vector<vector<double> > event;
    for (int i=0;i<coarse.size();i=i+3) {
        analyzeData(coarse[i],coarse[i+1],coarse[i+2],event);
    }
    /*******************/

    /** cluster ax, ay and rz separately **/
    int k = 2; // into 2 clusters
    vector<vector<double> > axs, ays, rzs;
    for (int i=0;i<event.size();i=i+3) {
        axs.push_back(event[i]);
        ays.push_back(event[i+1]);
        rzs.push_back(event[i+2]);
    }
    /* cluster pruning */
    vector<vector<double> > prunedAxs, prunedAys, prunedRzs;
    pruning(axs,2,prunedAxs);
    pruning(ays,2,prunedAys);
    pruning(rzs,2,prunedRzs);
    /*************************************/

    /** find topK templates **/
    int s, K;
    s = prunedAxs.size(); K = min(10,s);
    vector<vector<double> > topAxs = topKSim(prunedAxs,K);
    s = prunedAys.size(); K = min(10,s);
    vector<vector<double> > topAys = topKSim(prunedAys,K);
    s = prunedRzs.size(); K = min(10,s);
    vector<vector<double> > topRzs = topKSim(prunedRzs,K);

    templates.push_back(topAxs);
    templates.push_back(topAys);
    templates.push_back(topRzs);
}

vector<string> readFileList(string fname, int cols)
{
    int tstno;
    string rz, ax, ay;
    vector<string> namelist;
    ifstream ifs;
    ifs.open(filename);
    if (!ifs) {
        cout << "file not found: " << filename << endl;
        exit(-1);
    }
    while (!ifs.eof()) {
        ifs >> tstno >> rz >> ax >> ay;
        namelist.push_back(ax);
        namelist.push_back(ay);
        namelist.push_back(rz);
    }
    return namelist;
}

int main()
{

//    vector<int> data;
//    for (int i=0;i<10;++i) data.push_back(i);
//    cout << "data:";
//    copy(data.begin(),data.end(),ostream_iterator<int>(cout," "));
//    cout << endl;
//    vector<int> a,b;
//    copy(data.begin()+1,data.begin()+5,a.begin());
//    cout << "a:";
//    copy(a.begin(),a.end(),ostream_iterator<int>(cout," "));
//    cout << endl;
//    copy(data.begin()+6,data.begin()+8,b.begin());
//    cout << "a:";
//    copy(b.begin(),b.end(),ostream_iterator<int>(cout," "));
//    cout << endl;


    vector<string> Ffilenames;
    Ffilenames.push_back("./data/XAXIS/v10178_001aa0.tsv");
    Ffilenames.push_back("./data/XAXIS/v10179_001aa0.tsv");
    Ffilenames.push_back("./data/XAXIS/v11820_001aa0.tsv");
    Ffilenames.push_back("./data/XAXIS/v10189_001aa0.tsv");
    Ffilenames.push_back("./data/XAXIS/v10195_001aa0.tsv");
    Ffilenames.push_back("./data/XAXIS/v11819_001aa0.tsv");
    vector<vector<vector<double> > > Ftemplates;
    createTemplate(Ffilenames,Ftemplates);

//    cout << "Hello world!" << endl;
//    vector<double> signal1 = readData("v10178_001aa0.tsv",2,false);
//    vector<double> signal2 = readData("v10179_001aa0.tsv",2,false);
//    vector<double> signal3 = readData("v10184_001aa0.tsv",2,false);
//    vector<double> signal4 = readData("v10189_001aa0.tsv",2,false);
//    vector<double> signal5 = readData("v10195_001aa0.tsv",2,false);
//    vector<vector<double> > templates;
//    templates.push_back(signal1);
//    templates.push_back(signal2);
//    templates.push_back(signal3);
//    templates.push_back(signal4);
//    templates.push_back(signal5);
//
//    /// k-means clustering ///
//    int k = 2;
//    vector<vector<double> > centroids;
//    for (int i=0;i<k;++i)
//        centroids.push_back(templates[i]);
//    vector<int> belonging(templates.size(),-1);
//    kMeans(templates,k,centroids,belonging);
//    /// find cluster of smallest value distribution.
//    double minvalue = DBL_MAX;
//    int c = -1;
//    for (int i=0;i<centroids.size();++i) {
//        double value = 0;
//        int n = 0;
//        vector<double> center = centroids[i];
//        for (int j=0;j<templates.size();++j) {
//            if (belonging[j]==i) {
//                value += dtwDistance(center,templates[j]);
//                ++n;
//            }
//        }
//        value = value / n;
//        if (value < minvalue) {
//            minvalue = value;
//            c = i;
//        }
//    }
//    /// cluster c is the cluster with smallest value distribution.
//    cout << "cluster with smallest value distribution is " << c << endl;
//    vector<vector<double> > final_templates;
//    for (int i=0;i<templates.size();++i) {
//        if (belonging[i]==c) {
//            final_templates.push_back(templates[i]);
//        }
//    }
//    //////////////////////////
//
////    /// get top k similar out of template set ///
////    vector<vector<double> > topk = topKSim(templates,2);
////    /////////////////////////////////////////////
////
////    vector<double> event = readData("v10178_004aa0.tsv",2,false);
////    double dr = distanceRatio(event,templates);
////    cout << "distance ratio = " << dr << endl;

    return 0;
}
