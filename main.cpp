#pragma warning(disable:4819)
#pragma warning(disable:4244)
#pragma warning(disable:4267)

#include <time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <limits>
#include <vector>
#include<queue>
#include<list>

#define index(i,j,k) i+2*(j)+4*(k)
const uint m_threshold = 125;
enum type { ROOT, LEAF };

struct node {
	int x[2];
	int y[2];
	int z[2];
	type Type;
	long num;
	bool surface;
	node* prt;
	std::vector<node*>* chi;
	bool isFull() { return (x[1] - x[0]) * (y[1] - y[0]) * (z[1] - z[0]) == num; }
	node(int Xmax = 100, int Ymax = 100, int Zmax = 100, int Xmin = 0, int Ymin = 0, int Zmin = 0, type _type = ROOT)
	{
		x[1] = Xmax; x[0] = Xmin; y[1] = Ymax; y[0] = Ymin; z[1] = Zmax; z[0] = Zmin; Type = _type;
		chi = new std::vector<node*>(8, nullptr);
		num = 0;
		surface = false;
		prt = nullptr;
	}

	/*node *FindPoint(int x, int y, int z)
	{
		std::queue<node*> q;
		for (int i = 0; i < 8; i++)
		{
			if ((*chi)[i] != nullptr)
				q.push((*chi)[i]);
		}
		node* tempNode = (*chi)[0];
		while (q.size() != 0)
		{
			tempNode = q.front();
			q.pop();
			if (tempNode->Type == LEAF && tempNode->surface)
			{
				if (tempNode->x[0] == x && tempNode->y[0] == y && tempNode->z[0] == z)
					return tempNode;
			}
			else
			{
				for (int i = 0; i < 8; i++)
				{
					if ((*(tempNode->chi))[i] != nullptr)
						q.push((*(tempNode->chi))[i]);
				}
			}
		}
	}*/

	node *FindPoint(int x, int y, int z, node* node)
	{
		if (node->x[0] == x && node->y[0] == y && node->z[0] == z)
			return node;
		int dx = x < (node->x[0] + node->x[1]) / 2 ? 0 : 1;
		int dy = y < (node->y[0] + node->y[1]) / 2 ? 0 : 1;
		int dz = z < (node->z[0] + node->z[1]) / 2 ? 0 : 1;
		//std::cout << dx << dy << dz << std::endl;
		if ((*(node->chi))[index(dx, dy, dz)] == NULL)
			return NULL;
		return FindPoint(x, y, z, (*(node->chi))[index(dx, dy, dz)]);
	}
};

struct surNode {
	int co[3];
	Eigen::Vector3f normal;
	bool judge;
	std::list<surNode*> child;
	surNode(int cox = 0, int coy = 0, int coz = 0, double nx = 0, double ny = 0, double nz = 0, bool _judge = false) {
		co[0] = cox; co[1] = coy; co[2] = coz; normal[0] = nx; normal[1] = ny; normal[2] = nz; judge = _judge;
	}
};


// 用于判断投影是否在visual hull内部
struct Projection
{
	Eigen::Matrix<float, 3, 4> m_projMat;
	cv::Mat m_image;


	bool outOfRange(int x, int max)
	{
		return x < 0 || x >= max;
	}

	bool checkRange(double x, double y, double z)
	{
		Eigen::Vector3f vec3 = m_projMat * Eigen::Vector4f(x, y, z, 1);
		int indX = vec3[1] / vec3[2];
		int indY = vec3[0] / vec3[2];

		if (outOfRange(indX, m_image.size().height) || outOfRange(indY, m_image.size().width))
			return false;
		return m_image.at<uchar>((uint)(vec3[1] / vec3[2]), (uint)(vec3[0] / vec3[2])) > m_threshold;
	}

};

// 用于index和实际坐标之间的转换
struct CoordinateInfo
{
	int m_resolution;
	double m_min;
	double m_max;

	double index2coor(int index)
	{
		return m_min + index * (m_max - m_min) / m_resolution;
	}

	CoordinateInfo(int resolution = 10, double min = 0.0, double max = 100.0)
		: m_resolution(resolution)
		, m_min(min)
		, m_max(max)
	{
	}
};


class Model
{
public:
	typedef std::vector<std::vector<bool>> Pixel;
	typedef std::vector<std::vector<int>> Pixel1;
	typedef std::vector<Pixel> Voxel;
	typedef std::vector<Pixel1> Voxel1;

	Model(int resX = 100, int resY = 100, int resZ = 100);
	~Model();

	void saveModel(const char* pFileName);
	void saveModelWithNormal(const char* pFileName);
	void loadMatrix(const char* pFileName);
	void loadImage(const char* pDir, const char* pPrefix, const char* pSuffix);
	void getModel();
	void getSurface();
	Eigen::Vector3f getNormal(int indX, int indY, int indZ);

	//后加的

	void buildTree(int x, int y, int z);
	int depth, max;


private:
	CoordinateInfo m_corrX;
	CoordinateInfo m_corrY;
	CoordinateInfo m_corrZ;

	int m_neiborSize;

	std::vector<Projection> m_projectionList;

	Voxel m_voxel;
	Voxel m_surface;

	node* head;
	surNode* surHead;

	std::list<node*> m_surfaceList;
};

void Model::buildTree(int x, int y, int z)
{
	int Depth = 0;
	node* tNode1 = head, *tNode2;
	tNode1->num++;
	int i, j, k, xm, ym, zm, tcase, x0, x1, y0, y1, z0, z1;
	while (Depth != depth)
	{
		x0 = tNode1->x[0]; x1 = tNode1->x[1];
		y0 = tNode1->y[0]; y1 = tNode1->y[1];
		z0 = tNode1->z[0]; z1 = tNode1->z[1];
		xm = (x0 + x1) / 2;
		ym = (y0 + y1) / 2;
		zm = (z0 + z1) / 2;
		i = x < xm ? 0 : 1;
		j = y < ym ? 0 : 1;
		k = z < zm ? 0 : 1;
		tcase = index(i, j, k);
		switch (tcase)
		{
		case 0: {
			tNode2 = (*(tNode1->chi))[tcase];
			if (tNode2 == nullptr)
			{
				tNode2 = new node(xm, ym, zm, x0, y0, z0);
				(*(tNode1->chi))[tcase] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 1: {
			tNode2 = (*(tNode1->chi))[tcase];
			if (tNode2 == nullptr) {
				tNode2 = new node(x1, ym, zm, xm, y0, z0);
				(*(tNode1->chi))[tcase] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 2: {
			tNode2 = (*(tNode1->chi))[tcase];
			if (tNode2 == nullptr) {
				tNode2 = new node(xm, y1, zm, x0, ym, z0);
				(*(tNode1->chi))[tcase] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 3: {
			tNode2 = (*(tNode1->chi))[tcase];
			if (tNode2 == nullptr) {
				tNode2 = new node(x1, y1, zm, xm, ym, z0);
				(*(tNode1->chi))[tcase] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 4: {
			tNode2 = (*(tNode1->chi))[tcase];
			if (tNode2 == nullptr) {
				tNode2 = new node(xm, ym, z1, x0, y0, zm);
				(*(tNode1->chi))[tcase] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 5: {
			tNode2 = (*(tNode1->chi))[tcase];
			if (tNode2 == nullptr) {
				tNode2 = new node(x1, ym, z1, xm, y0, zm);
				(*(tNode1->chi))[tcase] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 6: {
			tNode2 = (*(tNode1->chi))[tcase];
			if (tNode2 == nullptr) {
				tNode2 = new node(xm, y1, z1, x0, ym, zm);
				(*(tNode1->chi))[tcase] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		case 7: {
			tNode2 = (*(tNode1->chi))[tcase];
			if (tNode2 == nullptr) {
				tNode2 = new node(x1, y1, z1, xm, ym, zm);
				(*(tNode1->chi))[tcase] = tNode2;
				tNode2->prt = tNode1;
			}
			break;
		}
		default:break;
		}
		tNode2->num++;
		tNode1 = tNode2;
		Depth++;
	}
	tNode1->Type = LEAF;
}

Model::Model(int resX, int resY, int resZ)
	: m_corrX(resX, -5, 5)
	, m_corrY(resY, -10, 10)
	, m_corrZ(resZ, 15, 30)
{
	if (resX > 100)
		m_neiborSize = resX / 100;
	else
		m_neiborSize = 1;
	m_voxel = Voxel(m_corrX.m_resolution, Pixel(m_corrY.m_resolution, std::vector<bool>(m_corrZ.m_resolution, true)));
	m_surface = m_voxel;
	//m_surface = Voxel1(m_corrX.m_resolution, Pixel1(m_corrY.m_resolution, std::vector<int>(m_corrZ.m_resolution, 0)));

	depth = 0;
	max = 1;
	while (max < resX || max < resY || max < resZ) {
		depth++; max *= 2;
	}
	head = new node(max, max, max);

}

Model::~Model()
{
}

void Model::saveModel(const char* pFileName)
{
	std::ofstream fout(pFileName);

	for (int indexX = 0; indexX < m_corrX.m_resolution; indexX++)
		for (int indexY = 0; indexY < m_corrY.m_resolution; indexY++)
			for (int indexZ = 0; indexZ < m_corrZ.m_resolution; indexZ++)
				if (!m_surface[indexX][indexY][indexZ])
				{
					double coorX = m_corrX.index2coor(indexX);
					double coorY = m_corrY.index2coor(indexY);
					double coorZ = m_corrZ.index2coor(indexZ);
					fout << coorX << ' ' << coorY << ' ' << coorZ << std::endl;
				}
}

void Model::saveModelWithNormal(const char* pFileName)
{
	std::ofstream fout(pFileName);

	//for遍历
	/*for (int indexX = 0; indexX < m_corrX.m_resolution; indexX++)
		for (int indexY = 0; indexY < m_corrY.m_resolution; indexY++)
			for (int indexZ = 0; indexZ < m_corrZ.m_resolution; indexZ++)
			{
				if (m_surface[indexX][indexY][indexZ] == 1)
				{
					double coorX = m_corrX.index2coor(indexX);
					double coorY = m_corrY.index2coor(indexY);
					double coorZ = m_corrZ.index2coor(indexZ);
					fout << coorX << ' ' << coorY << ' ' << coorZ << ' ';
					Eigen::Vector3f nor = getNormal(indexX, indexY, indexZ);
					fout << nor(0) << ' ' << nor(1) << ' ' << nor(2) << std::endl;
				}
			}*/

	//迭代器遍历
	for (auto it = m_surfaceList.cbegin(); it != m_surfaceList.cend(); it++) 
	{
		fout << m_corrX.index2coor((*it)->x[0]) << ' ' << m_corrY.index2coor((*it)->y[0]) << ' ' << m_corrZ.index2coor((*it)->z[0]) << ' ';
		Eigen::Vector3f nor = getNormal((*it)->x[0], (*it)->y[0], (*it)->z[0]);
		fout << nor(0) << ' ' << nor(1) << ' ' << nor(2) << std::endl;
	}
}

void Model::loadMatrix(const char* pFileName)
{
	std::ifstream fin(pFileName);

	int num;
	Eigen::Matrix<float, 3, 3> matInt;
	Eigen::Matrix<float, 3, 4> matExt;
	Projection projection;
	while (fin >> num)
	{
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				fin >> matInt(i, j);

		double temp;
		fin >> temp;
		fin >> temp;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 4; j++)
				fin >> matExt(i, j);

		projection.m_projMat = matInt * matExt;
		m_projectionList.push_back(projection);
	}
}

void Model::loadImage(const char* pDir, const char* pPrefix, const char* pSuffix)
{
	int fileCount = m_projectionList.size();
	std::string fileName(pDir);
	fileName += '/';
	fileName += pPrefix;
	for (int i = 0; i < fileCount; i++)
	{
		std::cout << fileName + std::to_string(i) + pSuffix << std::endl;
		m_projectionList[i].m_image = cv::imread(fileName + std::to_string(i) + pSuffix, CV_8UC1);
	}
}

void Model::getModel()
{
	int num = 0;
	int prejectionCount = m_projectionList.size();

	for (int indexX = 0; indexX < m_corrX.m_resolution; indexX++)
	{
		double coorX = m_corrX.index2coor(indexX);
		for (int indexY = 0; indexY < m_corrY.m_resolution; indexY++)
		{
			double coorY = m_corrY.index2coor(indexY);
			for (int indexZ = 0; indexZ < m_corrZ.m_resolution; indexZ++)
			{
				double coorZ = m_corrZ.index2coor(indexZ);
				for (int i = 0; i < prejectionCount; i++)
				{
					if (!(m_voxel[indexX][indexY][indexZ] = m_projectionList[i].checkRange(coorX, coorY, coorZ)))
						break;
				}
				if (m_voxel[indexX][indexY][indexZ])
				{
					buildTree(indexX, indexY, indexZ);
					num++;
				}
			}
		}
	}
	std::cout << "\n模型点数量：" << num << std::endl;
}

void Model::getSurface()
{
	// 邻域：上、下、左、右、前、后。
	int dx[6] = { -1, 0, 0, 0, 0, 1 };
	int dy[6] = { 0, 1, -1, 0, 0, 0 };
	int dz[6] = { 0, 0, 0, 1, -1, 0 };
	int num = 0;
	// lambda表达式，用于判断某个点是否在Voxel的范围内
	auto outOfRange = [&](int indexX, int indexY, int indexZ) {
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= m_corrX.m_resolution
			|| indexY >= m_corrY.m_resolution
			|| indexZ >= m_corrZ.m_resolution;
	};

	std::queue<node*> q;
	q.push(head);
	node* tNode;
	while (q.size() != 0)
	{
		tNode = q.front();
		q.pop();
		if (tNode->Type == LEAF)
		{
			bool ans = false;
			for (int i = 0; i < 6; i++)
			{
				ans = ans || outOfRange(tNode->x[0] + dx[i], tNode->y[0] + dy[i], tNode->z[0] + dz[i])
					|| !m_voxel[tNode->x[0] + dx[i]][tNode->y[0] + dy[i]][tNode->z[0] + dz[i]];
				if (ans)
				{
					m_surface[tNode->x[0]][tNode->y[0]][tNode->z[0]] = false;
					tNode->surface = true;
					m_surfaceList.push_back(tNode);
					num++;
					break;
				}
			}
		}
		else
		{

			if (tNode->num != 0 && !tNode->isFull())
			{
				for (int i = 0; i < 8; i++)
				{
					if ((*(tNode->chi))[i] != nullptr)
						q.push((*(tNode->chi))[i]);
				}
			}
		}

	}
	std::cout << "\n表面点数量：" << num << std::endl;
}

Eigen::Vector3f Model::getNormal(int indX, int indY, int indZ)
{
	auto outOfRange = [&](int indexX, int indexY, int indexZ)
	{
		return indexX < 0 || indexY < 0 || indexZ < 0
			|| indexX >= m_corrX.m_resolution
			|| indexY >= m_corrY.m_resolution
			|| indexZ >= m_corrZ.m_resolution;
	};

	std::vector<Eigen::Vector3f> neiborList;
	std::vector<Eigen::Vector3f> innerList;

	for (int dx = -m_neiborSize; dx <= m_neiborSize; dx++)
	{
		for (int dy = -m_neiborSize; dy <= m_neiborSize; dy++)
		{
			for (int dz = -m_neiborSize; dz <= m_neiborSize; dz++)
			{
				if (!dx && !dy && !dz)
					continue;
				int neiborX = indX + dx;
				int neiborY = indY + dy;
				int neiborZ = indZ + dz;
				if (!outOfRange(neiborX, neiborY, neiborZ))
				{
					node *neiborNode = head->FindPoint(neiborX, neiborY, neiborZ, head);
					if (neiborNode == NULL)
						continue;
					float coorX = m_corrX.index2coor(neiborX);
					float coorY = m_corrY.index2coor(neiborY);
					float coorZ = m_corrZ.index2coor(neiborZ);
					if (neiborNode->surface)
						neiborList.push_back(Eigen::Vector3f(coorX, coorY, coorZ));
					else
						innerList.push_back(Eigen::Vector3f(coorX, coorY, coorZ));
				}
			}
		}
	}


	Eigen::Vector3f point(m_corrX.index2coor(indX), m_corrY.index2coor(indY), m_corrZ.index2coor(indZ));

	Eigen::MatrixXf matA(3, neiborList.size());
	for (int i = 0; i < neiborList.size(); i++)
		matA.col(i) = neiborList[i] - point;
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigenSolver(matA * matA.transpose());
	Eigen::Vector3f eigenValues = eigenSolver.eigenvalues();
	int indexEigen = 0;
	if (abs(eigenValues[1]) < abs(eigenValues[indexEigen]))
		indexEigen = 1;
	if (abs(eigenValues[2]) < abs(eigenValues[indexEigen]))
		indexEigen = 2;
	Eigen::Vector3f normalVector = eigenSolver.eigenvectors().col(indexEigen);

	Eigen::Vector3f innerCenter = Eigen::Vector3f::Zero();
	for (auto const& vec : innerList)
		innerCenter += vec;
	innerCenter /= innerList.size();

	if (normalVector.dot(point - innerCenter) < 0)
		normalVector *= -1;
	return normalVector;
}

int main(int argc, char** argv)
{
	clock_t total_t = clock();

	// 分别设置xyz方向的Voxel分辨率
	clock_t t = clock();
	Model model(300, 300, 300);
	t = clock() - t;
	std::cout << "\ntime for initialization: " << (float(t) / CLOCKS_PER_SEC) << "seconds\n";

	// 读取相机的内外参数
	t = clock();
	model.loadMatrix("../../calibParamsI.txt");
	t = clock() - t;
	std::cout << "\ntime for loading matrix: " << (float(t) / CLOCKS_PER_SEC) << "seconds\n";

	// 读取投影图片
	t = clock();
	model.loadImage("../../wd_segmented", "WD2_", "_00020_segmented.png");
	t = clock() - t;
	std::cout << "\ntime for loading images: " << (float(t) / CLOCKS_PER_SEC) << "seconds\n";

	// 得到Voxel模型
	t = clock();
	model.getModel();
	t = clock() - t;
	std::cout << "get model done\n";
	std::cout << "time for getting model: " << (float(t) / CLOCKS_PER_SEC) << "seconds\n";

	// 获得Voxel模型的表面
	t = clock();
	model.getSurface();
	t = clock() - t;
	std::cout << "get surface done\n";
	std::cout << "time for getting surface: " << (float(t) / CLOCKS_PER_SEC) << "seconds\n";

	// 将模型导出为xyz格式
	t = clock();
	model.saveModel("../../WithoutNormal.xyz");
	t = clock() - t;
	std::cout << "\nsave without normal done\n";
	std::cout << "time for saving model without normal: " << (float(t) / CLOCKS_PER_SEC) << "seconds\n";

	t = clock();
	model.saveModelWithNormal("../../WithNormal.xyz");
	t = clock() - t;
	std::cout << "\nsave with normal done\n";
	std::cout << "time for saving model with normal: " << (float(t) / CLOCKS_PER_SEC) << "seconds\n";

	t = clock();
	system("PoissonRecon.x64 --in ../../WithNormal.xyz --out ../../mesh.ply");
	t = clock() - t;
	std::cout << "\nsave mesh.ply done\n";
	std::cout << "time for poinsson reconstruction " << (float(t) / CLOCKS_PER_SEC) << "seconds\n";

	total_t = clock() - total_t;
	std::cout << "time: " << (float(total_t) / CLOCKS_PER_SEC) << "seconds\n";

	//std::cout << count1 << ',' << count2;
	system("pause");
	return (0);
}