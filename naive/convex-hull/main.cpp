#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace std::chrono;

struct Point
{
    int x, y;
};

int side(Point p, Point q, Point r)
{
    int val = (q.y - p.y) * (r.x - q.x) -
              (q.x - p.x) * (r.y - q.y);
    if (val == 0) return 0;  // Collinear to p->q
    return (val > 0.0f) ? 1 : -1;  // Clockwise || Counterclockwise to p->q
}

vector<Point> giftWrapping(vector<Point> points)
{
    vector<Point> hull;

    if (points.size() < 3) return points;

    int leftMost = 0;
    for (unsigned i = 1; i < points.size(); i++)
        if (points[i].x < points[leftMost].x)
            leftMost = i;

    int p = leftMost, q;
    do
    {
        hull.push_back(points[p]);
        q = (p + 1) % points.size();
        for (unsigned i = 0; i < points.size(); i++)
        {
            if (side(points[p], points[i], points[q]) == -1)
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

    unsigned totalPoints;
    vector<Point> points;

    fin >> totalPoints;
    for (unsigned i = 0; i < totalPoints; i++) {
        int x, y;
        fin >> x >> y;
        Point p = {x, y};
        points.push_back(p);
    }
    fin.close();

    auto start = high_resolution_clock::now();
    vector<Point> hull = giftWrapping(points);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by function: " << duration.count() << " microseconds" << endl;

    ofstream fout("output.txt");

    for (auto it = hull.begin(); it != hull.end(); it++)
        fout << it->x << " " << it->y << endl;

    fout.close();

    return 0;
}