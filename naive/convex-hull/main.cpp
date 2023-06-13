#include <algorithm>
#include <vector>
#include <set>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace std::chrono;

typedef pair<int, int> Point;

int side(Point p, Point q, Point r)
{
    int val = (q.second - p.second) * (r.first - q.first) -
                (q.first - p.first) * (r.second - q.second);
    if (val == 0) return 0;  // Collinear to p->q
    return (val > 0.0f) ? 1 : 2;  // Clockwise or counterclockwise to p->q
}

set<Point> convexHull(vector<Point> points)
{
    set<Point> hull;

    if (points.size() < 3) return hull;

    int leftMost = 0;
    for (int i = 1; i < points.size(); i++)
        if (points[i].first < points[leftMost].first)
            leftMost = i;

    int p = leftMost, q;
    do
    {
        hull.insert(points[p]);
        q = (p + 1) % points.size();
        for (int i = 0; i < points.size(); i++)
        {
            if (side(points[p], points[i], points[q]) == 2)
                q = i;
        }
        p = q;
    } while (p != leftMost);

    return hull;
}

int main (int argc, char** argv) 
{

    if (argc != 2) 
    {
        cout << "Usage: ./naive <input_file>" << endl;
        return 1;
    }

    ifstream fin(argv[1]);

    int totalPoints;
    vector<Point> points;

    fin >> totalPoints;
    for (int i = 0; i < totalPoints; i++) {
        float x, y;
        fin >> x >> y;
        points.push_back(make_pair(x, y));
    }
    fin.close();

    auto start = high_resolution_clock::now();
    set<Point> hull = convexHull(points);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by function: " << duration.count() << " microseconds" << endl;

    ofstream fout("output.txt");
    for(auto it = hull.begin(); it != hull.end(); it++)
        fout << it->first << " " << it->second << endl;

    fout.close();

    return 0;
}