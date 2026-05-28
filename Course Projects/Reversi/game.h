void place() {

	int color, i, j;
	if (p)
		color = 1;
	else
		color = -1;

	if (!check(color)) {
		update(-color);
		//ai = 1;
		return;
	}

	//cout << "human\n";

	while (1) {
		M = getmessage();
		if (M.lbutton == 1) {
			if (M.x >= 0 && M.x < leng && M.y >= 0 && M.y < leng) {
				savecover();
				return;
			}

			if (M.x >= 0 && M.x < leng && M.y >= leng && M.y <= 2 * leng) {
				if (scount >= 2&&returncount<3){
					returncount++;
					scount -= 2;
				}
					
				for (i = 1; i <= 8; i++)
					for (j = 1; j <= 8; j++)
						chess[i][j] = save[0].chess[i][j][scount];
				save[0].step = scount;
				update(color);
			}

			x11 = (M.x - x00) / leng + 1, y11 = (M.y - x00) / leng + 1;
			if (chess[x11][y11] == 0 && acc(x11, y11, color)) {
				chess[x11][y11] = color;
				turn(x11, y11, color);
				update(-color);
				//ai = difficulty;
				break;
			}
		}
	}
	return;
}

void black() {

	switch (blackp) {
		case 0: {
			place();
			break;
		}

		case 1: {
			Sleep(500);
			aiplace_ez();
			break;
		}

		case 2: {
			Sleep(100);
			aiplace_medium();
			break;
		}

		case 3: {
			aiplace_hard();
			break;
		}
	}
	p = false;
	return;
}

void white() {

	switch (whitep) {
		case 0: {
			place();
			break;
		}

		case 1: {
			Sleep(500);
			aiplace_ez();
			break;
		}

		case 2: {
			Sleep(100);
			aiplace_medium();
			break;
		}

		case 3: {
			aiplace_hard();
			break;
		}
	}
	p = true;
	return;
}

void first() {
	returncount=0;
	for (int i = 1; i <= 8; i++)
		for (int j = 1; j <= 8; j++)
			chess[i][j] = 0;
	chess[4][4] = -1, chess[5][5] = -1, chess[5][4] = 1, chess[4][5] = 1;
	p = true;

	for (int i = 1; i <= 8; i++)
		for (int j = 1; j <= 8; j++)
			save[0].chess[i][j][scount] = chess[i][j];
	save[0].black = blackp;
	save[0].white = whitep;
	save[0].step = scount;
	return;
}

void start() {
	putimage(0, 0, &chessplate);

	if (p)
		update(1);
	else
		update(-1);
	while (!end()) {
		if (p) {
			//cout<<"black\n";
			black();
		}

		else if (!p) {
			//cout<<"white\n";
			white();
		}

		scount++;
		for (int i = 1; i <= 8; i++)
			for (int j = 1; j <= 8; j++)
				save[0].chess[i][j][scount] = chess[i][j];
		save[0].step = scount;

	}
	int black = 0, white = 0;
	for (int i = 1; i <= 8; i++)
		for (int j = 1; j <= 8; j++) {
			if (chess[i][j] == 1)
				black++;
			if (chess[i][j] == -1)
				white++;
		}

	if (black > white)
		drawAlpha(&blackwins, 50, 0);
	else if (black == white)
		cout << "DRAW\n";
	else
		drawAlpha(&whitewins, 50, 0);
	return;
}


bool search(int x, int y, int d, int color) {
	if (chess[x + dx[d]][y + dy[d]] == 0 || x == 0 || x == 9 || y == 0 || y == 9)
		return false;
	if (chess[x + dx[d]][y + dy[d]] == -color)
		return search(x + dx[d], y + dy[d], d, color);
	if (chess[x + dx[d]][y + dy[d]] == color)
		return true;
}


void turn(int x, int y, int color) {
	int xk, yk;
	for (int i = 0; i < 8; i++) {
		xk = x, yk = y;
		if (search(xk, yk, i, color)) {
			while (chess[xk + dx[i]][yk + dy[i]] == -color) {
				chess[xk + dx[i]][yk + dy[i]] = color;
				xk += dx[i], yk += dy[i];
			}
		}
	}
	return;
}


bool acc(int x, int y, int color) { //To judge the accesibility of a placement
	bool judge = false;
	for (int d = 0; d <= 7; d++)
		if (chess[x + dx[d]][y + dy[d]] == -color) {
			judge = search(x + dx[d], y + dy[d], d, color);
			if (judge)
				return true;
		}
//direction:0:west,1:north,2:east,3:south
	return false;
}


bool check(int color) {
	for (int i = 1; i <= 8; i++)
		for (int j = 1; j <= 8; j++)
			if (chess[i][j] == 0)
				if (acc(i, j, color))
					return true;

	return false;
}


bool end() {
	//int i, j, black = 0, white = 0;
	if (!check(1) && !check(-1))
		return true;

	return false;
}

