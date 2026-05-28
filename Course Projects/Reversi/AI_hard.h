bool MCTSsearch(node V, int x, int y, int d) {
	if (V.chess[x + dx[d]][y + dy[d]] == 0 || x == 0 || x == 9 || y == 0 || y == 9)
		return false;
	if (V.chess[x + dx[d]][y + dy[d]] == -V.color)
		return MCTSsearch(V, x + dx[d], y + dy[d], d);
	if (V.chess[x + dx[d]][y + dy[d]] == V.color)
		return true;
}

bool MCTSacc(node V, int x, int y) {
	bool judge = false;
	if (V.chess[x][y] != 0)
		return false;
	int d;
	for (d = 0; d <= 7; d++)
		if (V.chess[x + dx[d]][y + dy[d]] == -V.color) {
			judge = MCTSsearch(V, x + dx[d], y + dy[d], d);
			if (judge)
				return true;
		}

	return false;
}

bool MCTScheck(node V) {
	int i, j;
	for (i = 1; i <= 8; i++)
		for (j = 1; j <= 8; j++)
			if (V.chess[i][j] == 0)
				if (MCTSacc(V, i, j))
					return true;

	return false;
}

bool MCTSend(node V) {
	node VV = V;
	VV.color = -V.color;
	if (!MCTScheck(V) && !MCTScheck(VV))
		return true;
	else
		return false;
}


void aiplace_hard() {
	int color, i, j;
	int maxi, maxj, qq = 0;
	float maxsimu = -1;
	if (p)
		color = 1;
	else
		color = -1;

	aicolor = color;

	if (!check(color)) {
		update(-color);
		//ai = 0;
		return;
	}

	//cout << "ai\n";

	memset(&v, 0, sizeof(v));
	v[0].color = color, nodecnt = 0, SIMU = 0, v[0].father = -1;
	for (i = 1; i <= 8; i++)
		for (j = 1; j <= 8; j++)
			v[0].chess[i][j] = chess[i][j];

	while (SIMU < times) {
		MCTS();
		SIMU++;
	}
	//根据搜索结果进行决策
	float rate = 0;
	for (i = 0; i <= v[0].son.size() - 1; i++) {
		//cout << v[v[0].son[i]].x << ' ' << v[v[0].son[i]].y << ' ' << v[v[0].son[i]].qual << '/' << v[v[0].son[i]].simu <<
		    // " searching son\n";
		float winrate = (float)v[v[0].son[i]].qual / v[v[0].son[i]].simu;
		if (rate < winrate/*maxsimu<v[v[0].son[i]].simu*/) {
			rate = winrate;
			maxsimu = v[v[0].son[i]].simu;
			qq = v[v[0].son[i]].qual;
			maxi = v[v[0].son[i]].x, maxj = v[v[0].son[i]].y;
		}
	}
	cout << "winrate:" << fixed << setprecision(2) << rate * 100 << "%\n";
	//cout << maxi << ' ' << maxj << '\n';
	chess[maxi][maxj] = color;
	turn(maxi, maxj, color);
	update(-color);
	//ai = 0;
	return;
}

void MCTS() {//执行一次MCTS过程
	int placement = 0;
	bool simuresult;
	
	while (!expandable(v[placement]) && v[placement].son.size() > 0) {
		//cout<<v[placement].son.size()<<"sonsize\n";
		placement = selection(placement);
		//cout<<placement<<"selection\n";
	}
	//selection到一个未访问节点


	if (!MCTSend(v[placement]) && expandable(v[placement])) { //将该节点expansion(如果游戏在该节点还未结束)
		expansion(placement);//simulation并且backtrace
		simuresult = simulation(v[nodecnt]);
		backtrace(nodecnt, simuresult);
	} else {
		simuresult = simulation(v[placement]);
		backtrace(placement, simuresult);
	}

	return;
}

void expansion(int place) {
	int i, j, xx, yy, sons = 0, place1;
	bool placeable = false;
	for (i = 1; i <= 8; i++)
		for (j = 1; j <= 8; j++) {
			if (placeable)
				break;
			if (MCTSacc(v[place], i, j)) {
				if (sons == v[place].son.size()) { //每次创建一个新的节点
					placeable = true;
					//cout<<nodecnt<<"nodecnt\n";
					nodecnt++;
					v[nodecnt].father = place, v[nodecnt].x = i, v[nodecnt].y = j, v[nodecnt].color = -v[place].color;
					for (xx = 1; xx <= 8; xx++)
						for (yy = 1; yy <= 8; yy++)
							step[xx][yy] = v[place].chess[xx][yy];

					step[i][j] = v[place].color;
					stepturn(i, j, v[place].color);
					for (xx = 1; xx <= 8; xx++)
						for (yy = 1; yy <= 8; yy++)
							v[nodecnt].chess[xx][yy] = step[xx][yy];
					v[place].son.push_back(nodecnt);
				}
				sons++;
			}
		}

	if (!placeable) {
		nodecnt++;
		v[nodecnt].father = place, v[nodecnt].x = 0, v[nodecnt].y = 0, v[nodecnt].color = -v[place].color;
		for (xx = 1; xx <= 8; xx++)
			for (yy = 1; yy <= 8; yy++)
				v[nodecnt].chess[xx][yy] = v[place].chess[xx][yy];

		v[place].son.push_back(nodecnt);
		expansion(nodecnt);
	}
	return;
}

bool simulation(node V) { //模拟一盘残局,判断ai是否赢了
	int color = -V.color, i, j, maxi, maxj, mx = -INFI, k;
	for (i = 1; i <= 8; i++) {
		for (j = 1; j <= 8; j++) {
			step[i][j] = V.chess[i][j];
			//cout<<step[i][j]<<' ';
		}
		//cout<<'\n';
	}

	endend = 0;
	while (!stepend()) {
		color = -color, mx = -INFI - 1, endend = 1;
		//cout<<color<<'\n';
		if (!stepcheck(color))
			continue;

		//随机simulation
		srand(time(0));
		while (1) {
			i = (rand() % 8) + 1;
			j = (rand() % 8) + 1;
			if (step[i][j] == 0 && stepacc(i, j, color)) {
				step[i][j] = color;
				stepturn(i, j, color);
				//update(-color);
				break;
			}
		}
	}

	//判断winner
	int black = 0, white = 0;
	for (i = 1; i <= 8; i++)
		for (j = 1; j <= 8; j++) {
			if (step[i][j] == 1)
				black++;
			if (step[i][j] == -1)
				white++;
		}

	if ((black - white)*aicolor > 0)
		return true;
	else
		return false;

}

int selection(int place) {
	int i, maxi = 0;
	float umax = 0, u = 0;
	if (v[place].color == aicolor) {
		umax = 0;
		for (i = 0; i <= v[place].son.size() - 1; i++) {

			if (v[v[place].son[i]].simu == 0) {
				umax = INF, maxi = i;
			}

			else {
				u = UCT(v[place].son[i]);
				//if(place==0)cout<<v[0].son[i]<<' '<<v[v[0].son[i]].qual<<'/'<<v[v[0].son[i]].simu<<' '<<u<<'\n';
				if (u > umax) {
					umax = u, maxi = i;
				}
			}
		}
	}

	else {
		umax = INF;
		for (i = 0; i <= v[place].son.size() - 1; i++) {

			if (v[v[place].son[i]].simu == 0) {
				umax = -INF, maxi = i;
			}

			else {
				u = UCT(v[place].son[i]);
				if (u < umax) {
					umax = u, maxi = i;
				}
			}
		}
	}
	return v[place].son[maxi];
}

void backtrace(int place, bool win) {
	v[place].simu++, v[place].qual += win;
	if (v[place].father != -1)
		backtrace(v[place].father, win);
	return;
}

float UCT(int place) {
	//float c = 1;
	//cout<<place<<' '<<(float)v[place].qual/v[place].simu<<'\n';
	return (float)v[place].qual / v[place].simu + c * sqrt(log(v[v[place].father].simu) / v[place].simu)
	       - cplace * pvalue[v[place].x][v[place].y] * aicolor * v[place].color;
}

bool expandable(node V) { //判断节点是否可被再次扩展
	int i, j, sons = 0;
	for (i = 1; i <= 8; i++)
		for (j = 1; j <= 8; j++)
			if (MCTSacc(V, i, j))
				sons++;
	if (sons > V.son.size())
		return true;
	else
		return false;
}
