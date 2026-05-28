bool p = true, pve = true, back = false, bgm = false;
const int x00 = 92.76, y00 = 92.76, leng = 90.5, xco = 570, yco = 74, lenco = 120,slen=170,xread=570,yread=360;
const int xset=60,yset=400;
const int INFI = 999999, times = 12000;
const float INF = 999999, c = 1.48,cplace=0.046;
int chess[10][10], depth, step[10][10], alpha, beta, SIMU, nodecnt,blackp,whitep; //1:black,0:null,-1:white
int value[10][10],xprior[61],yprior[61],pvalue[9][9];//定义位置优先级,用以位置优先策略
int dx[8] = {-1, 0, 1, 0, -1, -1, 1, 1}, dy[8] = {0, -1, 0, 1, 1, -1, 1, -1};
int ai = 0, x11, y11, xai, yai, cnt = 0,aicolor, endend = 0;
int difficulty=3,scount=0,returncount=0;
IMAGE chessplate, wchess, bchess, whitewins, blackwins, cover1, cover2, backbutton, available;
IMAGE whiteai,blackai,whitehuman,blackhuman,savecover1,saveread1,traceback,settings,setcover;
IMAGE ezon,ezoff,mdon,mdoff,hdon,hdoff,bgmon,bgmoff;
ExMessage M;

//为MCTS算法准备的节点记录
struct node {
	int chess[10][10];
	int color;//此时轮到color走子
	int x, y; //(x,y)为该节点落子位置
	int qual, simu; //qual为该节点胜利次数,simu为该节点模拟次数
	vector<int> son;//子节点位置
	int father;//父节点位置
}v[200001];

struct file{
	int chess[10][10][128];//file.chess[i][j][k]表示在k步后棋盘(i,j)处的状态(跳步算一步)
	int black,white;//记录对战者,0表示玩家,1为ai_ez,2为ai_medium,3为ai_hard
	int step;//表明对局进行了几步
}save[4];
//save[0]是cache,save[1/2/3]是存档1/2/3

//声明函数部分
void drawAlpha(IMAGE *picture, int  picture_x, int picture_y);//抠除透明图形的边框 
void update();//更新棋盘
bool search(int x, int y, int d, int color);//找寻d方向是否能"夹"
bool stepsearch(int x, int y, int d, int color);//对step[i][j]的是否能"夹"的判断
bool MCTSsearch(node V, int x, int y, int d);//对节点V中的判断
void turn(int x, int y, int color);//落子后翻转中间的棋子
void stepturn(int x, int y, int color);//对step[i][j]的翻转
bool acc(int x, int y, int color);//判断(x,y)处color能否落子
bool MCTSacc(node V, int x, int y);//对节点V中的判断
bool stepacc(int x, int y, int color);//对step[i][j]中的判断
bool check(int color);//判断color是否有地方落子
bool stepcheck(int color);//对step[i][j]中的判断
bool MCTScheck(node V);//对节点V中的判断
bool end();//判断游戏是否结束
bool stepend();//判断在step中是否结束
bool MCTSend(node V);//判断在节点V中是否结束
int turnamount(int x, int y, int color);//计算color在(x,y)处落子后的总收益(用于贪心AI)
int stepamount(int x, int y, int color);//计算step的落子后总收益(用于min-max搜索AI)
void aiplace_ez();//ez版本AI落子(贪心)
void aiplace_medium();//medium版本AI落子(min-max搜索)
int aisearch(int color, int depth, int alpha, int beta);//max层搜索(color走子颜色,depth搜索深度,alpha&beta用于剪枝)
int playersearch(int color, int depth, int alpha, int beta);//min层搜索
int evaluate(int color);//局面价值评估函数
void aiplace_hard();//hard版本AI落子(MCTS)
void MCTS();//管控蒙特卡洛过程的进行
void expansion(int place);//扩展一个叶子节点
bool simulation(node V);//对一个节点V进行模拟,随机进行一局棋局,记录胜者
int selection(int place);//根据UCT函数选择一个子节点
void backtrace(int place, bool win);//回溯,向上更新回溯结果(每个节点的simu&qual)直到根节点
float UCT(int place);//UCT函数,判断每个节点成为最优节点的期望值(影响因素:节点胜率,鼓励探索项,位置优势奖励项)
bool expandable(node V);//判断节点V是否可扩展
void place();//玩家落子
void black();//分配执黑落子权
void white();//分配执白落子权
void start();//开始游戏
void cover();//封面界面
void savecover();//询问存档界面
void first();//开始新游戏之前对棋盘的初始化
void saveread(int num);//读盘
void setting();//设置界面
void fileout();//写入存档文件 
void filein();//读取存档文件 

void initialize()
{
	int width = 900, height = 900, size = 100;
	
	initgraph(width, height);
	loadimage(&chessplate, _T("chessplate.jpeg"), 900, 900, true);
	loadimage(&bchess, _T("bchess.png"), leng, leng, true);
	loadimage(&wchess, _T("wchess.png"), leng, leng, true);
	loadimage(&whitewins, _T("whitewins.png"), 800, 100, true);
	loadimage(&blackwins, _T("blackwins.png"), 800, 100, true);
	loadimage(&cover1, _T("cover1.png"), 900, 900, true);
	loadimage(&cover2, _T("cover2.png"), 2 * lenco, 2 * lenco, true);
	loadimage(&backbutton, _T("backbutton.png"), leng, leng, true);
	loadimage(&available, _T("available.png"), leng, leng, true);
	loadimage(&blackai,_T("blackai.png"),60,60,true);
	loadimage(&whiteai,_T("whiteai.png"),60,60,true);
	loadimage(&blackhuman,_T("blackhuman.png"),60,60,true);
	loadimage(&whitehuman,_T("whitehuman.png"),60,60,true);
	loadimage(&savecover1,_T("savecover.png"),900,900,true);
	loadimage(&saveread1,_T("saveread.png"),2*lenco,2*lenco,true);
	loadimage(&traceback,_T("traceback.png"),leng,leng,true);
	loadimage(&settings,_T("settings.png"),leng,leng,true);
	loadimage(&setcover,_T("setcover.png"),900,900,true);
	loadimage(&ezon,_T("ezon.png"),leng,leng,true);
	loadimage(&ezoff,_T("ezoff.png"),leng,leng,true);
	loadimage(&mdon,_T("mdon.png"),leng,leng,true);
	loadimage(&mdoff,_T("mdoff.png"),leng,leng,true);
	loadimage(&hdon,_T("hdon.png"),leng,leng,true);
	loadimage(&hdoff,_T("hdoff.png"),leng,leng,true);
	loadimage(&bgmon,_T("bgmon.png"),leng,leng,true);
	loadimage(&bgmoff,_T("bgmoff.png"),leng,leng,true);

	value[1][1]=12,value[1][2]=-2,value[1][3]=3,value[1][4]=3,value[1][5]=3,value[1][6]=3,value[1][7]=-2,value[1][8]=12;
	value[2][1]=-2,value[2][2]=-3,value[2][3]=1,value[2][4]=1,value[2][5]=1,value[2][6]=1,value[2][7]=-3,value[2][8]=-2;
	value[3][1]=3,value[3][2]=1,value[3][3]=5,value[3][4]=5,value[3][5]=5,value[3][6]=5,value[3][7]=1,value[3][8]=3;
	value[4][1]=3,value[4][2]=1,value[4][3]=5,value[4][4]=7,value[4][5]=7,value[4][6]=5,value[4][7]=1,value[4][8]=3;
	value[5][1]=3,value[5][2]=1,value[5][3]=5,value[5][4]=7,value[5][5]=7,value[5][6]=5,value[5][7]=1,value[5][8]=3;
	value[6][1]=3,value[6][2]=1,value[6][3]=5,value[6][4]=5,value[6][5]=5,value[6][6]=5,value[6][7]=1,value[6][8]=3;
	value[7][1]=-2,value[7][2]=-3,value[7][3]=1,value[7][4]=1,value[7][5]=1,value[7][6]=1,value[7][7]=-3,value[7][8]=-2;
	value[8][1]=12,value[8][2]=-2,value[8][3]=3,value[8][4]=3,value[8][5]=3,value[8][6]=3,value[8][7]=-2,value[8][8]=12;
	
	pvalue[1][1]=4,pvalue[1][2]=0,pvalue[1][3]=2,pvalue[1][4]=2,pvalue[1][5]=2,pvalue[1][6]=2,pvalue[1][7]=0,pvalue[1][8]=4;
	pvalue[2][1]=0,pvalue[2][2]=0,pvalue[2][3]=1,pvalue[2][4]=1,pvalue[2][5]=1,pvalue[2][6]=1,pvalue[2][7]=0,pvalue[2][8]=0;
	pvalue[3][1]=2,pvalue[3][2]=1,pvalue[3][3]=3,pvalue[3][4]=3,pvalue[3][5]=3,pvalue[3][6]=3,pvalue[3][7]=1,pvalue[3][8]=2;
	pvalue[4][1]=2,pvalue[4][2]=1,pvalue[4][3]=3,pvalue[4][4]=0,pvalue[4][5]=0,pvalue[4][6]=3,pvalue[4][7]=1,pvalue[4][8]=2;
	pvalue[5][1]=2,pvalue[5][2]=1,pvalue[5][3]=3,pvalue[5][4]=0,pvalue[5][5]=0,pvalue[5][6]=3,pvalue[5][7]=1,pvalue[5][8]=2;
	pvalue[6][1]=2,pvalue[6][2]=1,pvalue[6][3]=3,pvalue[6][4]=3,pvalue[6][5]=3,pvalue[6][6]=3,pvalue[6][7]=1,pvalue[6][8]=2;
	pvalue[7][1]=0,pvalue[7][2]=0,pvalue[7][3]=1,pvalue[7][4]=1,pvalue[7][5]=1,pvalue[7][6]=1,pvalue[7][7]=0,pvalue[7][8]=0;
	pvalue[8][1]=4,pvalue[8][2]=0,pvalue[8][3]=2,pvalue[8][4]=2,pvalue[8][5]=2,pvalue[8][6]=2,pvalue[8][7]=0,pvalue[8][8]=4;
	
	return;
}

void fileout()
{
	freopen("save.txt","w",stdout);
	for(int i=0;i<=9;i++)
		for(int j=0;j<=9;j++)
			for(int k=0;k<=80;k++)cout<<save[1].chess[i][j][k]<<' ';
	
	cout<<save[1].black<<' '<<save[1].white<<' '<<save[1].step<<' ';
	//fclose(stdout); 
	
	for(int i=0;i<=9;i++)
		for(int j=0;j<=9;j++)
			for(int k=0;k<=80;k++)cout<<save[2].chess[i][j][k]<<' ';
	
	cout<<save[2].black<<' '<<save[2].white<<' '<<save[2].step<<' ';
	//fclose(stdout); 
	
	for(int i=0;i<=9;i++)
		for(int j=0;j<=9;j++)
			for(int k=0;k<=80;k++)cout<<save[3].chess[i][j][k]<<' ';
	
	cout<<save[3].black<<' '<<save[3].white<<' '<<save[3].step;
	//fclose(stdout); 
	freopen("CON","w",stdout);
	return;
}

void filein()
{
	freopen("save.txt","r",stdin);
	for(int i=0;i<=9;i++)
		for(int j=0;j<=9;j++)
			for(int k=0;k<=80;k++)cin>>save[1].chess[i][j][k];
	
	cin>>save[1].black>>save[1].white>>save[1].step;
	//fclose(stdin); 
	
	for(int i=0;i<=9;i++)
		for(int j=0;j<=9;j++)
			for(int k=0;k<=80;k++)cin>>save[2].chess[i][j][k];
	
	cin>>save[2].black>>save[2].white>>save[2].step;
	//fclose(stdin);
	
	for(int i=0;i<=9;i++)
		for(int j=0;j<=9;j++)
			for(int k=0;k<=80;k++)cin>>save[3].chess[i][j][k];
	
	cin>>save[3].black>>save[3].white>>save[3].step;
	//fclose(stdin);
	
	return;
}
