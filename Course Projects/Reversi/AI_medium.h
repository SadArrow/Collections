bool stepsearch(int x, int y, int d, int color) {
	if (step[x + dx[d]][y + dy[d]] == 0 || x == 0 || x == 9 || y == 0 || y == 9)
		return false;
	if (step[x + dx[d]][y + dy[d]] == -color)
		return stepsearch(x + dx[d], y + dy[d], d, color);
	if (step[x + dx[d]][y + dy[d]] == color)
		return true;
}


void stepturn(int x, int y, int color) {
	int xk, yk;
	for (int i = 0; i < 8; i++) {
		xk = x, yk = y;
		if (stepsearch(xk, yk, i, color)) {
			while (step[xk + dx[i]][yk + dy[i]] == -color) {
				step[xk + dx[i]][yk + dy[i]] = color;
				xk += dx[i], yk += dy[i];
			}
		}
	}
	return;
}

bool stepacc(int x, int y, int color) { //To judge the accesibility of a placement
	bool judge = false;
	int d;
	for (d = 0; d <= 7; d++)
		if (step[x + dx[d]][y + dy[d]] == -color) {
			judge = stepsearch(x + dx[d], y + dy[d], d, color);
			if (judge)
				return true;
		}

//direction:0:west,1:north,2:east,3:south
	return false;
}

bool stepcheck(int color) {
	int i, j;
	for (i = 1; i <= 8; i++)
		for (j = 1; j <= 8; j++)
			if (step[i][j] == 0)
				if (stepacc(i, j, color))
					return true;

	return false;
}

bool stepend() {
	//int i, j, black = 0, white = 0;
	if (!stepcheck(1) && !stepcheck(-1))
		return true;

	return false;
}


int stepamount(int x, int y, int color) {
	cnt = 0;
	int d;
	for (d = 0; d <= 7; d++) {
		int xm = x, ym = y, cntt = 0;
		while (step[xm + dx[d]][ym + dy[d]] == -color)
			xm += dx[d], ym += dy[d], cntt++;
		if (step[xm + dx[d]][ym + dy[d]] == color && xm + dx[d] != 0 && xm + dx[d] != 9 && ym + dy[d] != 0 && ym + dy[d] != 9)
			cnt += cntt;
	}
	return cnt;
}

void aiplace_medium() {
	int color, i, j, maxi, maxj;
	int val, mx = -INFI - 1;
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

	for (i = 1; i <= 8; i++)
		for (j = 1; j <= 8; j++)
			step[i][j] = chess[i][j];

	depth = 6, alpha = -INFI, beta = INFI;
	for (i = 1; i <= 8; i++)
		for (j = 1; j <= 8; j++) {
			if (step[i][j] == 0 && stepacc(i, j, color)) {
				step[i][j] = color;
				int dline[8];
				memset(dline, 0, sizeof(dline));
				int xk, yk;
				for (int ii = 0; ii < 8; ii++) {
					xk = i, yk = j;
					if (stepsearch(xk, yk, ii, color)) {
						while (step[xk + dx[ii]][yk + dy[ii]] == -color) {
							step[xk + dx[ii]][yk + dy[ii]] = color;
							xk += dx[ii], yk += dy[ii];
							dline[ii]++;
						}
					}
				}//stepturn
				val = playersearch(-color, depth - 1, alpha, beta);
				if (val > mx) {
					mx = val;
					maxi = i;
					maxj = j;
				}
				step[i][j] = 0;
				for (int ii = 0; ii < 8; ii++)
					for (int jj = 0; jj < dline[ii]; jj++)
						step[i + dx[ii] * (jj + 1)][j + dy[ii] * (jj + 1)] = -color;
				//stepreturn
			}
		}

	cout << val << '\n';
	chess[maxi][maxj] = color;
	turn(maxi, maxj, color);
	update(-color);
	//ai = 0;
	return;

}

int aisearch(int color, int depth, int alpha, int beta) {
	int mx = -INFI - 1, mn = INFI + 1, sr;
	int i, j;
	bool stepable = false;
	if (depth <= 0 || stepend()) {
		//cout<<alpha<<' '<<beta<<'\n';
		return evaluate(aicolor);
	}

	for (i = 1; i <= 8; i++)
		for (j = 1; j <= 8; j++) {
			if (step[i][j] == 0 && stepacc(i, j, -color)) {
				stepable = true;
				step[i][j] = color;
				int dline[8];
				memset(dline, 0, sizeof(dline));

				int xk, yk;
				for (int ii = 0; ii < 8; ii++) {
					xk = i, yk = j;
					if (stepsearch(xk, yk, ii, color)) {
						while (step[xk + dx[ii]][yk + dy[ii]] == -color) {
							step[xk + dx[ii]][yk + dy[ii]] = color;
							xk += dx[ii], yk += dy[ii];
							dline[ii]++;
						}
					}
				}//stepturn
				sr = playersearch(-color, depth - 1, alpha, beta);
				mx = max(mx, sr);
				step[i][j] = 0;

				for (int ii = 0; ii < 8; ii++)
					for (int jj = 0; jj < dline[ii]; jj++)
						step[i + dx[ii] * (jj + 1)][j + dy[ii] * (jj + 1)] = -color;
				//stepreturn

				if (sr > beta)
					return beta;
				if (sr > alpha)
					alpha = sr;
			}
		}
	if (!stepable) {
		sr = aisearch(-color, depth - 1, alpha, beta);
		mx = sr;
		if (sr > beta)
			return beta;
		if (sr > alpha)
			alpha = sr;
	}
	return mx;
	//if(depth%2==0)return mn;
}

int playersearch(int color, int depth, int alpha, int beta) {
	int mx = -INFI - 1, mn = INFI + 1, sr;
	int i, j;
	bool stepable = false;
	if (depth <= 0 || stepend()) {
		//cout<<alpha<<' '<<beta<<'\n';
		return evaluate(aicolor);
	}

	for (i = 1; i <= 8; i++)
		for (j = 1; j <= 8; j++) {
			if (step[i][j] == 0 && stepacc(i, j, -color)) {
				stepable = true;
				step[i][j] = color;
				int dline[8];
				memset(dline, 0, sizeof(dline));

				int xk, yk;
				for (int ii = 0; ii < 8; ii++) {
					xk = i, yk = j;
					if (stepsearch(xk, yk, ii, color)) {
						while (step[xk + dx[ii]][yk + dy[ii]] == -color) {
							step[xk + dx[ii]][yk + dy[ii]] = color;
							xk += dx[ii], yk += dy[ii];
							dline[ii]++;
						}
					}
				}//stepturn
				sr = aisearch(-color, depth - 1, alpha, beta);
				mn = min(mn, sr);
				step[i][j] = 0;

				for (int ii = 0; ii < 8; ii++)
					for (int jj = 0; jj < dline[ii]; jj++)
						step[i + dx[ii] * (jj + 1)][j + dy[ii] * (jj + 1)] = -color;
				//stepreturn

				if (sr < beta)
					beta = sr;
				if (sr < alpha)
					return alpha;
			}
		}
	if (!stepable) {
		sr = aisearch(-color, depth - 1, alpha, beta);
		mn = sr;
		if (sr < beta)
			beta = sr;
		if (sr < alpha)
			return alpha;
	}
	//if(depth%2==1)return mx;
	return mn;
}

int evaluate(int color) {
	int score = 0;
	int i, j, winning = 0;
	for (i = 1; i <= 8; i++)
		for (j = 1; j <= 8; j++) {
			score += step[i][j] * color + value[i][j] * step[i][j] * color;
			winning += step[i][j] * color;
		}
	bool endif = stepend();
	if (endif && winning > 0)
		return INFI;
	else if (endif && winning < 0)
		return -INFI;
	else
		return score;
}