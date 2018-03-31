#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <iterator>
#include <float.h>
#include <cmath>
#include <climits>
#include <assert.h>

#define LEFT -1
#define DIAGONAL 0
#define UP 1
#define NIL INT_MAX

using namespace std;
typedef int DIRECTION;

vector<double> readData(const string &filename)
{
    ifstream file;
    file.open(filename);
    if (!file) {
        cerr << "file not found: " << filename << endl;
        exit(-1);
    }
    vector<double> signal;
    while (!file.eof()) {
        double t,s;
        file >> t >> s;
        signal.push_back(s);
    }
    return signal;
}

double dist(double a, double b)
{
    return fabs(a-b);
    // return (a-b) * (a-b);
}

double dtwDistance(const vector<double> &signal1, const vector<double> &signal2)
{
    int n1 = signal1.size();
    int n2 = signal2.size();
    double **dtw = new double*[n1+1];
    for (int i=0;i<n1+1;++i) dtw[i] = new double[n2+1];

    for (int i=0;i<n1+1;++i) dtw[i][0] = DBL_MAX;
    for (int i=0;i<n2+1;++i) dtw[0][i] = DBL_MAX;
    dtw[0][0] = 0;

    for (int i=1;i<n1+1;++i) {
        for (int j=1;j<n2+1;++j) {
            double cost = dist(signal1[i-1], signal2[j-1]);
            dtw[i][j] = cost + min(dtw[i-1][j],min(dtw[i][j-1],dtw[i-1][j-1]));
        }
    }
    return dtw[n1][n2];
}

DIRECTION argmin_Direction(double diag, double up, double left)
{
    if (diag > up) {
        if (up > left)
            return LEFT;
        else
            return UP;
    }
    else {
        if (diag > left)
            return LEFT;
        else
            return DIAGONAL;
    }
}

double barycenter(const vector<double> &tab)
{
    double sum = 0.0;
    for (int i=0;i<tab.size();++i)
        sum += tab[i];
    return sum / tab.size();
}

void sequenceAverage(const vector<vector<double> > sequences, vector<double> &center)
{
    int T = sequences[0].size();
    int T_ = center.size();
//    double **assocTab = new double*[T_];
//    for (int i=0;i<T_;++i)
//        assocTab[i] = new double[sequences.size()];
    vector<vector<double> > assocTab(T_);

    // the cost of warping path;
    double **costMatrix = new double*[T_];
    for (int i=0;i<T_;++i)
        costMatrix[i] = new double[T];
    // the exact path trace;
    int **pathMatrix = new int*[T_];
    for (int i=0;i<T_;++i)
        pathMatrix[i] = new int[T];
    // optimal path length;
    int **optPathLength = new int*[T_];
    for (int i=0;i<T_;++i)
        optPathLength[i] = new int[T];

    for (int s=0;s<sequences.size();++s) {
        vector<double> seq = sequences[s];
        costMatrix[0][0] = dist(center[0],seq[0]);
        pathMatrix[0][0] = NIL;
        optPathLength[0][0] = 0;

        for (int i=1;i<T_;++i) {
            costMatrix[i][0] = costMatrix[i-1][0] + dist(center[i],seq[0]);
            pathMatrix[i][0] = UP;
            optPathLength[i][0] = i;
        }
        for (int j=1;j<T;++j) {
            costMatrix[0][j] = costMatrix[0][j-1] + dist(seq[j],center[0]);
            pathMatrix[0][j] = LEFT;
            optPathLength[0][j] = j;
        }
        double res;
        for (int i=1;i<T_;++i) {
            for (int j=1;j<T;++j) {
                DIRECTION d = argmin_Direction(costMatrix[i-1][j-1],costMatrix[i-1][j],costMatrix[i][j-1]);
                pathMatrix[i][j] = d;
                switch (d) {
                case DIAGONAL:
                    res = costMatrix[i-1][j-1];
                    optPathLength[i][j] = optPathLength[i-1][j-1] + 1;
                    break;
                case LEFT:
                    res = costMatrix[i][j-1];
                    optPathLength[i][j] = optPathLength[i][j-1] + 1;
                    break;
                case UP:
                    res = costMatrix[i-1][j];
                    optPathLength[i][j] = optPathLength[i-1][j] + 1;
                    break;
                }
                costMatrix[i][j] = res + dist(center[i],seq[j]);
            }
        }
        int nbTuplesAverageSeq = optPathLength[T-1][T-1] + 1;
        int i, j;
        i = j = T - 1;
        for (int t=nbTuplesAverageSeq-1;t>=0;--t) {
            assocTab[i].push_back(seq[j]);
            switch (pathMatrix[i][j]) {
            case DIAGONAL:
                i--;
                j--;
                break;
            case LEFT:
                j--;
                break;
            case UP:
                i--;
                break;
            }
        }
    }

    for (int t=0;t<T_;++t) {
        center[t] = barycenter(assocTab[t]);
    }
}

void Kmeans(const vector<vector<double> > &sequences, int k, vector<vector<double> > &centroids, vector<int> &belonging)
{
    assert(k==centroids.size());
    assert(sequences.size()==belonging.size());
    int T = sequences[0].size();
    int N = sequences.size();

    for (int i=0;i<N;++i) belonging[i] = 0; // 初始认为所有点属于第一个cluster。

    bool change = true;
    int numIter = 0;
    while (change) {
        change = false;
        for (int i=0;i<N;++i) {
            double mindist = DBL_MAX;
            int c = -1;
            for (int j=0;j<k;++j) {
                double d = dtwDistance(sequences[i],centroids[j]);
                if (d < mindist) {
                    mindist = d;
                    c = j;
                }
            }
            if (c != belonging[i] && c != -1) {
                change = true;
                belonging[i] = c;
            }
        }
        // cout << "belonging: "; copy(belonging.begin(),belonging.end(),ostream_iterator<int>(cout," ")); cout << endl;
        for (int j=0;j<k;++j) {
            vector<vector<double> > cluster;
            for (int i=0;i<N;++i) {
                if (belonging[i] == j)
                    cluster.push_back(sequences[i]);
            }
            assert(cluster.size()>0);
            if (cluster.size()>1) {
                vector<double> center = cluster[0];
                sequenceAverage(cluster,center);
                centroids[j] = center;
            }
            else {
                centroids[j] = cluster[0];
            }
        }
    }
}

int smallestCluster(const vector<vector<double> > &sequences, const vector<vector<double> > &centroids, const vector<int> &belonging)
{
    double minvalue = DBL_MAX;
    int c = -1;
    for (int i=0;i<centroids.size();++i) {
        double value = 0;
        int n = 0;
        vector<double> center = centroids[i];
        for (int j=0;j<sequences.size();++j) {
            if (belonging[j] == i) {
                value += dtwDistance(center,sequences[j]);
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

void pruning(const vector<vector<double> > &sequences, int k, vector<vector<double> > &templates, int maxNumTemplates)
{
    vector<vector<double> > centroids;
    for (int i=0;i<k;++i)
        centroids.push_back(sequences[i]);
    vector<int> belonging(sequences.size(),0);
    Kmeans(sequences,k,centroids,belonging);
    int c = smallestCluster(sequences,centroids,belonging);
    for (int i=0;i<sequences.size();++i) {
        if (belonging[i] == c) {
            if (templates.size()<maxNumTemplates)
                templates.push_back(sequences[i]);
            else {
                // check if sequences[i] should be put into templates;
                double maxdist = -DBL_MAX;
                int c = -1;
                double sum = 0;
                for (int j=0;j<templates.size();++j) {
                    for (int k=0;k<templates.size();++k) {
                        if (j!=k) {
                            sum += dtwDistance(templates[j],templates[k]);
                        }
                    }
                    if (sum > maxdist) {
                        maxdist = sum;
                        c = j;
                    }
                }
                sum = 0;
                for (int j=0;j<templates.size();++j) {
                    if (j!=c) {
                        sum += dtwDistance(sequences[i],templates[j]);
                    }
                }
                if (sum < maxdist) {
                    templates[c] = sequences[i];
                }
            }
        }
    }
}

vector<string> readFileList(string fname)
{
    int tstno;
    string rz, ax, ay;
    vector<string> namelist;
    ifstream ifs;
    ifs.open(fname);
    if (!ifs) {
        cout << "file not found: " << fname << endl;
        exit(-1);
    }
    while (!ifs.eof()) {
        ifs >> tstno >> rz >> ax >> ay;
        namelist.push_back("data/XAXIS/"+ax);
        namelist.push_back("data/YAXIS/"+ay);
        namelist.push_back("data/ROTAT/"+rz);
    }
    return namelist;
}

void loadData(const vector<string> &filenames, vector<vector<double> > &sequences)
{
    int numFiles = filenames.size();
    for (int i=0;i<numFiles;i=i+3) {
        sequences.push_back(readData(filenames[i]));
        sequences.push_back(readData(filenames[i+1]));
        sequences.push_back(readData(filenames[i+2]));
    }
}

void test()
{
    vector<double> data1 = readData("v10178_001aa0.tsv");
    vector<double> data2 = readData("v10178_004aa0.tsv");
    vector<double> data3 = readData("v10179_001aa0.tsv");

    vector<double> seq1(1000), seq2(1000), seq3(1000);
    copy(data1.begin(),data1.begin()+1000,seq1.begin());
    copy(data2.begin(),data2.begin()+1000,seq2.begin());
    copy(data3.begin(),data3.begin()+1000,seq3.begin());

    vector<vector<double> > sequences;
    sequences.push_back(seq1);
    sequences.push_back(seq2);
    sequences.push_back(seq3);

    vector<vector<double> > templates;
    pruning(sequences,2,templates,10);
}

int main()
{
    //test();
    /* Main function shows how to create templates for front impact */
    /**
    Main function shows how to create templates for front impact
    "front-impact-templates.txt" has a format like:
    [TSTNO]   [Rz-file] [Ax-file] [Ay-file]
    */
    cout << "Hello world!" << endl;

    vector<string> frontImapctFiles = readFileList("front-impact-templates.txt");
    vector<vector<double> > sequences;
    loadData(frontImapctFiles,sequences);
    vector<vector<double> > ax,ay,rz;
    for (int i=0;i<sequences.size();i=i+3) {
        rz.push_back(sequences[i]);
        ax.push_back(sequences[i+1]);
        ay.push_back(sequences[i+2]);
    }
    vector<vector<double> > axTemplates, ayTemplates, rzTemplates;
    pruning(rz,2,rzTemplates,10);
    pruning(ax,2,axTemplates,10);
    pruning(ay,2,ayTemplates,10);

    return 0;
}
