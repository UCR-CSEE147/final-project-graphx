#include <iostream>
#include <algorithm>
#include <vector>
#include <execution>
#include <chrono>
#include <fstream>

using namespace std;
using namespace std::chrono;

struct Point
{
    int x, y;
};

int side (Point p, Point q, Point r)
{
    int val = (q.y - p.y) * (r.x - q.x) -
              (q.x - p.x) * (r.y - q.y);
 
    if (val == 0) return 0;
    return (val > 0) ? 1 : -1; 
}

vector<Point> giftWrapping(vector<Point> points)
{
    if (points.size() < 3) return points;
 
    vector<Point> hull;
 
    // Find the leftmost point
    int l = distance(points.begin(), min_element(std::execution::par, points.begin(), points.end(), 
        [](Point a, Point b) {
            return a.x < b.x;
        }));
 
    int p = l, q;
    do
    {
        hull.push_back(points[p]);
 
        q = (p+1)%points.size();
        for (unsigned i = 0; i < points.size(); i++)
        {
           if (side(points[p], points[i], points[q]) == -1)
               q = i;
        }
 
        p = q;
 
    } while (p != l); 

    return hull;
}

int main (int argc, char** argv) 
{

    if (argc != 2) 
    {
        cout << "Usage: ./moderncpp <input_file>" << endl;
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
