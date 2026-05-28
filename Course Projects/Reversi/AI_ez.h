void aiplace_ez() {
	int color, ii, jj, maxi, maxj;
	int mx = 0;
	if (p)
		color = 1;
	else
		color = -1;

	if (!check(color)) {
		update(-color);
		ai = 0;
		return;
	}
	//EZ AI
	for (ii = 1; ii <= 8; ii++)
		for (jj = 1; jj <= 8; jj++)
			if (chess[ii][jj] == 0 && acc(ii, jj, color) && mx < turnamount(ii, jj, color) + value[ii][jj]) {
				mx = turnamount(ii, jj, color) + value[ii][jj];
				maxi = ii, maxj = jj;
			}

	chess[maxi][maxj] = color;
	turn(maxi, maxj, color);
	update(-color);
	//ai = 0;

	return;
}

int turnamount(int x, int y, int color) { //统计能吃多少子
	cnt = 0;
	int d;
	for (d = 0; d <= 7; d++) {
		int xm = x, ym = y, cntt = 0;
		while (chess[xm + dx[d]][ym + dy[d]] == -color)
			xm += dx[d], ym += dy[d], cntt++;
		if (chess[xm + dx[d]][ym + dy[d]] == color && xm + dx[d] != 0 && xm + dx[d] != 9 && ym + dy[d] != 0 && ym + dy[d] != 9)
			cnt += cntt;
	}
	return cnt;
}