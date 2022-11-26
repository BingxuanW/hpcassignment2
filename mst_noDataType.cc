#include <bits/stdc++.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include "mpi.h"
#include <iostream>
#include <fstream>

using namespace std;

/* Original graph from .mtx */
static vector<tuple<int, int, double>> Graph;     // Read .mtx and fill it , <row, coloum, weight>
static int VerticeNum = 0;                        // Number of vertice
static int EdgeNum = 0;                           // Number of edges

/* About process */
static vector<double> p_subtree_total;     // Combination of results of all processes
static vector<double> p_subtree_final;     // Further integration
static vector<tuple<int, int, int, double>> tree_final;     // < root index, row, column ,weight>
static vector<tuple<int, int, double>> Graph_subtree_total;
/* load data from .mtx */
void loadGraph(const char *file, vector<tuple<int, int, double>> &Graph);

/* About Boruvkas */
int find(vector<pair<int, int>> &trees, int i);
void Union(vector<pair<int, int>> &trees, int a, int b);
void Boruvkas(vector<pair<int, int>> &trees, vector<tuple<int, int, double>> Graph, int V, int E, vector<double> &p_subtree);
bool comp(const tuple<int, int, int, double> &a, tuple<int, int, int, double> &b);


/***************************    main()   **************************/
int main(int argc, char **argv)
{
  if (argc != 2)
  {
    fprintf(stderr, "usage: %s <filename>\n", argv[0]);
    return -1;
  }

  MPI_Init(&argc, &argv);

  int pid;                                       // process id
  int process_total;                             // number of processes
  vector<tuple<int, int, double>> Graph_p;       // Graph decompostion as input for each process
  //vector<tuple<int, int, double>> p_subtree;     // sub-tree created by each process
  vector<double>p_subtree;
  loadGraph(argv[1], Graph);    // load data from .mtx

  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &process_total);
  MPI_Barrier(MPI_COMM_WORLD);  // wait for everything to be ready

  int EdgeNum_per_process = ceil(EdgeNum / process_total) ;    // number of tasks assigned to each process (except the last one)
  int EdgeNum_last_process = EdgeNum - EdgeNum_per_process * (process_total - 1);   // number of tasks assigned to the last process


  /***********************************************************/
  double start_time = MPI_Wtime();
  /***********************************************************/


  /**********************       1. Assign edges to each process        *********************/
  if (pid == process_total - 1)
  {
    /* the last process */
    for (int i = 0; i < EdgeNum_last_process; i++)
    {
      Graph_p.push_back( Graph[ EdgeNum_per_process * pid + i ]) ;
    }
  }else{
    /* each process except the last */
    for (int i = 0; i < EdgeNum_per_process; i++)
    {
      Graph_p.push_back( Graph[ EdgeNum_per_process * pid + i ] );
    }
  }


  MPI_Barrier(MPI_COMM_WORLD);
  vector<pair<int, int>> trees;
  /*  Input:  Graph decompostion
      Output: &subtree from each process  */
  Boruvkas(trees, Graph_p, VerticeNum, Graph_p.size(), p_subtree);
  MPI_Barrier(MPI_COMM_WORLD);


  /******************    2. Gather calculation results of each process   *****************/
  int p_subtree_size = p_subtree.size();
  int *counts_recv = (int *)malloc(process_total * sizeof(int));  // counts of receive
  int *displs = (int *)malloc(process_total * sizeof(int));       // displacements
  

  if (pid == 0)
  {
    /* main process */
    MPI_Gather(&p_subtree_size, 1, MPI_INT, counts_recv, 1, MPI_INT, 0, MPI_COMM_WORLD);
    displs[0] = 0;
    for (int i = 1; i < process_total; i++)
    {
 	 displs[i] = displs[i - 1] + counts_recv[i - 1];
    }
  }
  else
  {
    MPI_Gather(&p_subtree_size, 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
  }

  //cout << "test 1"<<endl;
  /******************    3. Centralised processing after gathering   *****************/
  /* A new MPI_Datatype for <int,int,double> */
  // MPI_Datatype nodeType, nodeType2;
  // int blocklens[2]={2,1};
  // MPI_Aint lb, extent;
  // MPI_Aint displacements[] = {0, 8};
  // MPI_Datatype oldType[] = { MPI_INT, MPI_DOUBLE};
  // MPI_Type_create_struct(2, blocklens, displacements, oldType, &nodeType);
  // MPI_Type_get_extent( nodeType, &lb, &extent );
  // MPI_Type_create_resized( nodeType, lb, extent, &nodeType2 );
  // MPI_Type_commit(&nodeType2);

  //p_subtree_total.resize(counts_recv[process_total - 1] + displs[process_total - 1]);
  

  if (pid == 0)
  {
    /* main process receive info */
    p_subtree_total.resize(counts_recv[process_total - 1] + displs[process_total - 1]);
    MPI_Gatherv(&p_subtree.front(), p_subtree_size, MPI_DOUBLE , &p_subtree_total.front(), counts_recv, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  else
  {
    MPI_Gatherv(&p_subtree.front(), p_subtree_size, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  
 
  //cout << "test 2"<<endl;
  /******************    4. Main progress final integration   *****************/
  if (pid == 0)
  {
    for (int i =0 ; i < p_subtree_total.size(); i+=3)
    {
      Graph_subtree_total.push_back(make_tuple(p_subtree_total[i], p_subtree_total[i+1], p_subtree_total[i+2]));
    }

    vector<pair<int, int>> trees2;
    Boruvkas(trees2, Graph_subtree_total, VerticeNum, Graph_subtree_total.size(), p_subtree_final);
    for (int i = 0; i < p_subtree_final.size(); i+=3)
    {
      int rootIndex = find(trees2, p_subtree_final[i]);
      tree_final.push_back(make_tuple(rootIndex, p_subtree_final[i], p_subtree_final[i + 1], p_subtree_final[i + 2]));
    }
    
    sort(tree_final.begin(), tree_final.end(), comp);
   // cout << "test 3"<<endl;
   
    vector<int> edgeNum_tree;   // number of edges of each tree
    int temp = get<0>(tree_final[0]);
    int edge_count=1;
    for (int i = 1; i < tree_final.size(); i++)
    {
      if ( get<0>(tree_final[i]) == temp )
      {
        edge_count++;
      }
      else
      { // i.e. not the same root
        edgeNum_tree.push_back(edge_count);
        temp = get<0>(tree_final[i]);
        edge_count = 1;
      }
    }
    edgeNum_tree.push_back(edge_count);

    /*******************************************************/
    double end_time = MPI_Wtime();
    /*******************************************************/


    /***************************    5. Output   **************************/
    edge_count = 0;
    //int disconnect_count = 0;
    for (int i = 0; i < edgeNum_tree.size(); i++ )
    {
      fprintf(stdout, "\nnumber of edges in spanning_tree : %d\n", edgeNum_tree[i]);
      
      double total_tree_weight = 0.0;
      for (int j = 0; j < edgeNum_tree[i]; j++ )
      {
        if(j<4){  // print the first 4 edges of each spanning tree
          fprintf(stdout, "%d %d %.20f\n", get<1>(tree_final[edge_count + j]) , get<2>(tree_final[edge_count + j]), get<3>(tree_final[edge_count + j]) );
        }
        total_tree_weight += get<3>(tree_final[edge_count + j]);
      }
     // disconnect_count ++;
      edge_count += edgeNum_tree[i];
      fprintf(stdout, "total tree weight : %.20f\n\n", total_tree_weight);
    }
    //cout << "disconect spanning tree:"<< disconnect_count << endl;
    fprintf(stdout, "Computation time : %.20f\n", end_time - start_time);

  }

  MPI_Finalize();

  return 0;
}

/***************************    customized functions   **************************/
/* load data from .mtx */
void loadGraph(const char *file, vector<tuple<int, int, double>> &Graph)
{
  // ifstream input(file);
  ifstream infile;
  infile.open(file);
  string Line;
  int first_line_flag = 0;

  if (!infile.is_open())
  {
    printf("Error opening .mtx ! \n");
  }

  while (getline(infile, Line))
  {
    // if start with "%", skip
    if (Line[0] == '%')
    {
      continue;
    }
    else
    {
      first_line_flag++;
    }

    // first line is like : [ columnNum rowNum nonZeroNum ]
    if (first_line_flag == 1)
    {
      // number of vertice can be derived directly from the first line (excl. comments) from .mtx
      VerticeNum = stoi(Line.substr(0, Line.find_first_of(" ")));
    }
    else
    {
      // index of two " "
      int first_split_pos, second_split_pos;

      // Line is like : row column value
      if ((first_split_pos = Line.find(" ")) != string::npos)
      {
        int column = stoi(Line.substr(0, first_split_pos));

        int second_split_pos = Line.find_first_of(" ", first_split_pos + 1);
        int row = stoi(Line.substr(first_split_pos + 1, second_split_pos - first_split_pos));

        double value = stod(Line.substr(second_split_pos + 1, Line.length() - second_split_pos - 1));

        // Exclusion of the pair of row = column
        if (row != column)
        {
          Graph.push_back(make_tuple(row, column, value));
          EdgeNum++;
        }
      }
    }
  }
  printf("Initial Graph : Vertice Number = %d, Edge Number = %d \n", VerticeNum, EdgeNum);
  infile.close();
}

int find(vector<pair<int, int>> &trees, int i)
{
  // find root and make root as parent of i
  if (trees[i].second != i)
  {
    trees[i].second = find(trees, trees[i].second);
  }
  return trees[i].second;
}

void Union(vector<pair<int, int>> &trees, int a, int b)
{
  int rootA = find(trees, a);
  int rootB = find(trees, b);

  if (trees[rootA].first < trees[rootB].first)
  {
    trees[rootA].second = rootB;
  }
  else if (trees[rootA].first > trees[rootB].first)
  {
    trees[rootB].second = rootA;
  }
  else
  {
    trees[rootB].second = rootA;
    trees[rootA].first++;
  }
}

void Boruvkas(vector<pair<int, int>> &trees, vector<tuple<int, int, double>> Graph, int V, int E, vector<double> &p_subtree)
{
  for (int i = 0; i < V; i++)
  {
    trees.push_back(make_pair(0, i));
  }
  int TotalTrees = V;
  double MST_total_weight = 0;
  int flag = 1;

  while (flag)
  {
    flag = 0;
    vector<int> smallest_edge(V, -1);
    for (int i = 0; i < E; i++)
    {
      int setA = find(trees, get<0>(Graph[i]));
      int setB = find(trees, get<1>(Graph[i]));

      if (setA == setB)
        continue;
      else
      {
        if (smallest_edge[setA] == -1 || get<2>(Graph[smallest_edge[setA]]) > get<2>(Graph[i]))
          smallest_edge[setA] = i;

        if (smallest_edge[setB] == -1 || get<2>(Graph[smallest_edge[setB]]) > get<2>(Graph[i]))
          smallest_edge[setB] = i;
      }
    }

    // add eges to MST
    for (int i = 0; i < V; i++)
    {
      if (smallest_edge[i] != -1)
      {
        int setA = find(trees, get<0>(Graph[smallest_edge[i]]));
        int setB = find(trees, get<1>(Graph[smallest_edge[i]]));

        if (setA == setB)
          continue;
        flag = 1;
        MST_total_weight += get<2>(Graph[smallest_edge[i]]);

        // p_subtree.push_back(make_tuple(get<0>(Graph[smallest_edge[i]]),
        //                                    get<1>(Graph[smallest_edge[i]]),
        //                                    get<2>(Graph[smallest_edge[i]])));
        p_subtree.push_back(double(get<0>(Graph[smallest_edge[i]])));
        p_subtree.push_back(double(get<1>(Graph[smallest_edge[i]])));
        p_subtree.push_back(double(get<2>(Graph[smallest_edge[i]])));

        Union(trees, setA, setB);
        TotalTrees--;
      }
    }
  }
}

bool comp(const tuple<int, int, int, double> &a, tuple<int, int, int, double> &b)
{
  return get<0>(a) < get<0>(b);
}
