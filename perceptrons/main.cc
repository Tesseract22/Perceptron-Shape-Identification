#include <fstream>
#include <iostream>
#include <random>
#include <string>

#define WIDTH 100
#define HEIGHT 100

#define SIZE_LIMIT 2
#define EDGE_OFFSET 2

#define BATCH_SIZE 100
#define TRAIN_TIME 100
#define EPOCH 3
#define BIAS 0
#define LEARNING_RATE 0.25f

#define WEIGHT_PATH "debug/weights"
#define MODEL "model"
int iter = 0;
int seed = 54;

typedef float matrix[HEIGHT][WIDTH];
;

using namespace std;

matrix weights;
matrix drawer;
void ClearDrawer() {
  for (size_t y = 0; y < HEIGHT; ++y) {
    for (size_t x = 0; x < WIDTH; ++x) {
      drawer[y][x] = 0;
    }
  }
}

int Clampi(int a, int upper, int lower) {
  if (a > upper) return upper;
  if (a < lower) return lower;
  return a;
}

float FowardPass() {
  float res = 0;
  for (size_t y = 0; y < HEIGHT; ++y) {
    for (size_t x = 0; x < WIDTH; ++x) {
      res += weights[y][x] * drawer[y][x];
    }
  }
  return res;
}

void BackPass(bool excite) {
  float mal = (excite ? 1.0f : -1.0f) * LEARNING_RATE;
  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      weights[y][x] += (drawer[y][x] * mal);
    }
  }
}

void DrawRect(int x0, int y0, int w, int h, float fill) {
  ClearDrawer();
  if (x0 < 0 || x0 >= WIDTH || y0 < 0 || y0 >= HEIGHT) return;
  int left = w / 2;
  int right = w - left;
  int down = h / 2;
  int up = h - down;
  int y1 = Clampi(y0 - up, HEIGHT - 1, 0);
  int y2 = Clampi(y0 + down - 1, HEIGHT - 1, 0);
  int x1 = Clampi(x0 - left, WIDTH - 1, 0);
  int x2 = Clampi(x0 + right - 1, WIDTH - 1, 0);
  for (int y = y1; y <= y2; ++y) {
    for (int x = x1; x <= x2; ++x) {
      drawer[y][x] = fill;
    }
  }
}

void DrawCircle(int xc, int yc, int r, float fill) {
  ClearDrawer();
  if (xc < 0 || xc >= WIDTH || yc < 0 || yc >= HEIGHT) return;
  if (r < 1) return;
  r = xc - r >= 0 ? r : xc;
  r = xc + r < WIDTH ? r : WIDTH - xc - 1;
  r = yc - r >= 0 ? r : yc;
  r = yc + r < HEIGHT ? r : HEIGHT - yc - 1;
  for (int y = yc - r; y <= yc + r; ++y) {
    for (int x = xc - r; x <= xc + r; ++x) {
      int dy = y - yc;
      int dx = x - xc;
      if (dx * dx + dy * dy <= r * r) drawer[y][x] = fill;
    }
  }
}
void ToPPM(const string& path, matrix& m) {
  const int kColorOffsetR = 150;
  const int kColorOffsetG = 0;
  const int kColorOffsetB = 150;

  const float kColorRate = 200.0f;
  const int kColorRange = 255;

  ofstream fs(path);
  fs << "P3" << '\n';
  fs << WIDTH << ' ' << HEIGHT << '\n';
  fs << kColorRange << '\n';

  float mn = MAXFLOAT;
  float mx = -MAXFLOAT;
  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      mx = max(mx, m[y][x]);
      mn = min(mn, m[y][x]);
    }
  }

  for (int y = 0; y < HEIGHT; ++y) {
    for (int x = 0; x < WIDTH; ++x) {
      int new_pix = int(m[y][x] / (mx - mn) * kColorRate);
      fs << kColorOffsetR + new_pix << ' ' << kColorOffsetG << ' '
         << kColorOffsetB - new_pix;
      fs << '\n';
    }
    // cout << path << " to ppm successful" << endl;
  }
}
void RandRect() {
  int y0 = rand() % (HEIGHT - 2 * EDGE_OFFSET) + EDGE_OFFSET;
  int x0 = rand() % (WIDTH - 2 * EDGE_OFFSET) + EDGE_OFFSET;
  int h = rand() % (HEIGHT - SIZE_LIMIT) + SIZE_LIMIT;
  int w = rand() % (WIDTH - SIZE_LIMIT) + SIZE_LIMIT;
  float fill = rand() % 256;

  DrawRect(x0, y0, w, h, fill);
}

void RandCircle() {
  int yc = rand() % (HEIGHT - 2 * EDGE_OFFSET) + EDGE_OFFSET;
  int xc = rand() % (WIDTH - 2 * EDGE_OFFSET) + EDGE_OFFSET;
  int r = rand() % (min(WIDTH, HEIGHT) / 2) + SIZE_LIMIT;
  float fill = rand() % 256;

  DrawCircle(xc, yc, r, fill);
}

double Predict() {
  int loss = 0;
  srand(time(nullptr));
  for (int i = 0; i < BATCH_SIZE; ++i) {
    RandRect();
    float prediction = FowardPass();
    if (prediction > BIAS) {
      loss++;
    }
  }
  for (int i = 0; i < BATCH_SIZE; ++i) {
    RandCircle();
    float prediction = FowardPass();
    if (prediction <= BIAS) {
      loss++;
      BackPass(true);
    }
  }
  double correctness = 1.0 - (double(loss) / (2 * BATCH_SIZE));
  return correctness;
}

int Train() {
  // const int kR = 54;
  int loss = 0;
  srand(seed);
  for (int i = 0; i < BATCH_SIZE; ++i) {
    RandRect();
    float prediction = FowardPass();
    if (prediction > BIAS) {
      loss++;
      BackPass(false);
    }
  }
  srand(seed);
  for (int i = 0; i < BATCH_SIZE; ++i) {
    RandCircle();
    float prediction = FowardPass();
    if (prediction <= BIAS) {
      loss++;
      BackPass(true);
    }
  }
  return loss;
}
void TrainCycle() {
  for (int i = 0; i < TRAIN_TIME; ++i) {
    int loss = Train();
    cout << "Iteration: " << iter << ", loss: " << loss << endl;
    ToPPM(WEIGHT_PATH + to_string(iter) + ".ppm", weights);
    iter++;
    if (loss == 0) return;
  }
}

void TrainUntil() {
  bool converge = false;
  while (!converge) {
    int loss = Train();
    cout << "Iteration: " << iter << ", loss: " << loss << endl;
    converge = loss == 0;
    ToPPM(WEIGHT_PATH + to_string(iter) + ".ppm", weights);
    iter++;
  }
}

void SaveMatrix(const string& path, matrix& m) {
  ofstream fs(path);
  fs << m;
}

// void ReadMatrix(const string& path) {
//   ofstream fs(path);
//   matrix m;
//   fs >> m;
// }
void TrainEpoch() {
  for (int i = 0; i < EPOCH; ++i) {
    TrainCycle();
    seed++;
  }
}
int main() {
  TrainEpoch();
  cout << Predict() << endl;
}
