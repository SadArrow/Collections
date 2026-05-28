void drawAlpha(IMAGE *picture, int  picture_x, int picture_y) { //抠除透明图形的边框

	// 变量初始化
	DWORD *dst = GetImageBuffer();    // GetImageBuffer()函数，用于获取绘图设备的显存指针，EASYX自带
	DWORD *draw = GetImageBuffer();
	DWORD *src = GetImageBuffer(picture); //获取picture的显存指针
	int picture_width = picture->getwidth(); //获取picture的宽度，EASYX自带
	int picture_height = picture->getheight(); //获取picture的高度，EASYX自带
	int graphWidth = getwidth();       //获取绘图区的宽度，EASYX自带
	int graphHeight = getheight();     //获取绘图区的高度，EASYX自带
	int dstX = 0;    //在显存里像素的角标

	// 实现透明贴图 公式： Cp=αp*FP+(1-αp)*BP ， 贝叶斯定理来进行点颜色的概率计算
	for (int iy = 0; iy < picture_height; iy++) {
		for (int ix = 0; ix < picture_width; ix++) {
			int srcX = ix + iy * picture_width; //在显存里像素的角标
			int sa = ((src[srcX] & 0xff000000) >> 24); //0xAArrggbb;AA是透明度
			int sr = ((src[srcX] & 0xff0000) >> 16); //获取RGB里的R
			int sg = ((src[srcX] & 0xff00) >> 8);   //G
			int sb = src[srcX] & 0xff;              //B
			if (ix >= 0 && ix <= graphWidth && iy >= 0 && iy <= graphHeight && dstX <= graphWidth * graphHeight) {
				dstX = (ix + picture_x) + (iy + picture_y) * graphWidth; //在显存里像素的角标
				int dr = ((dst[dstX] & 0xff0000) >> 16);
				int dg = ((dst[dstX] & 0xff00) >> 8);
				int db = dst[dstX] & 0xff;
				draw[dstX] = ((sr * sa / 255 + dr * (255 - sa) / 255) <<
				              16)  //公式： Cp=αp*FP+(1-αp)*BP  ； αp=sa/255 , FP=sr , BP=dr
				             | ((sg * sa / 255 + dg * (255 - sa) / 255) << 8)         //αp=sa/255 , FP=sg , BP=dg
				             | (sb * sa / 255 + db * (255 - sa) / 255);              //αp=sa/255 , FP=sb , BP=db
			}
		}
	}
}

void update(int color) { //更新棋盘
	int white = 0, black = 0;
	putimage(0, 0, &chessplate);
	drawAlpha(&backbutton, 0, 0);
	drawAlpha(&traceback, 0, leng);
	//settextcolor(RGB(255,255,255));
	if (blackp == 0)
		drawAlpha(&blackhuman, 0, leng * 9 + 7);
	else
		drawAlpha(&blackai, 0, leng * 9 + 7);

	if (whitep == 0)
		drawAlpha(&whitehuman, leng - 5, leng * 9 + 7);
	else
		drawAlpha(&whiteai, leng - 5, leng * 9 + 7);


	for (int i = 1; i <= 8; i++)
		for (int j = 1; j <= 8; j++) {
			if (chess[i][j] == 0 && acc(i, j, color))
				drawAlpha(&available, x00 + leng * (i - 1), y00 + leng * (j - 1));

			if (chess[i][j] == 1)
				//putimage(x00 + leng * (i - 1), y00 + leng * (j - 1), &bchess);
				drawAlpha(&bchess, x00 + leng * (i - 1), y00 + leng * (j - 1));


			if (chess[i][j] == -1)
				drawAlpha(&wchess, x00 + leng * (i - 1), y00 + leng * (j - 1));
			//putimage(x00 + leng * (i - 1), y00 + leng * (j - 1), &wchess);
		}
	return;
}

void cover() {
	putimage(0, 0, &cover1);
	putimage(xco, yco, &cover2);
	putimage(xread, yread, &saveread1);
	drawAlpha(&settings, xset, yset);
	fileout();
	filein();
	
	//putimage(leng,leng,&whiteai);
	/*
	for(int n=1;n<=3;n++){
		cout<<n<<"save\n";
		//for(int ss=0;ss<=save[n].step;ss++){
			//cout<<ss<<'\n';
			cout<<save[n].step<<"depth\n";
			for(int i=1;i<=8;i++){
			for(int j=1;j<=8;j++)cout<<save[n].chess[i][j][save[n].step]<<' ';
			cout<<'\n';
			}
		}*/
	
	while (1) {
		M = getmessage();
		if (M.lbutton == 1) {
			if (M.x >= xco && M.x <= xco + lenco && M.y >= yco && M.y <= yco + lenco) {
				//pve = true, ai = difficulty;
				blackp = difficulty, whitep = 0, scount = 0;
				first();
				start();
				Sleep(5000);
				cover();
			}

			else if (M.x > xco + lenco && M.x <= xco + 2 * lenco && M.y >= yco && M.y <= yco + lenco) {
				//pve = true, ai = 0;
				blackp = 0, whitep = difficulty, scount = 0;
				first();
				start();
				Sleep(5000);
				cover();
			}

			else if (M.x >= xco && M.x <= xco + lenco && M.y > yco + lenco && M.y <= yco + 2 * lenco) {
				//pve = false;
				blackp = 0, whitep = 0, scount = 0;
				first();
				start();
				Sleep(5000);
				cover();
			}

			else if (M.x >= xset && M.x <= xset + leng && M.y >= yset && M.y <= yset + leng) {
				setting();
			}

			else if (M.x > xco + lenco && M.x <= xco + 2 * lenco && M.y > yco + lenco && M.y <= yco + 2 * lenco)
				exit(1);

			else if (M.x > xread + lenco && M.x <= xread + 2 * lenco && M.y >= yread && M.y <= yread + lenco)
				saveread(1);

			else if (M.x >= xread && M.x <= xread + lenco && M.y > yread + lenco && M.y <= yread + 2 * lenco)
				saveread(2);

			else if (M.x > xread + lenco && M.x <= xread + 2 * lenco && M.y > yread + lenco && M.y <= yread + 2 * lenco)
				saveread(3);


		}
	}
	return;
}

void savecover() {
	putimage(0, 0, &savecover1);
	while (1) {
		M = getmessage();
		if (M.lbutton == 1) {
			//cout<<M.x<<' '<<M.y<<"\n";
			if (M.x >= 98 && M.y >= 260 && M.x <= 98 + slen && M.y <= 260 + slen) {
				save[1] = save[0];
				//cout<<"save1\n";
				cover();
			} else if (M.x >= 340 && M.y >= 260 && M.x <= 340 + slen && M.y <= 260 + slen) {
				save[2] = save[0];
				//cout<<"save2\n";
				cover();
			} else if (M.x >= 620 && M.y >= 260 && M.x <= 620 + slen && M.y <= 260 + slen) {
				save[3] = save[0];
				//cout<<"save3\n";
				cover();
			} else if (M.x >= 340 && M.y >= 470 && M.x <= 340 + slen && M.y <= 470 + slen)
				cover();
		}
	}
	return;
}

void saveread(int num) {
	int st;
	st = save[num].step;
	cout << "saveread" << num << '\n';
	if (save[num].chess[4][4][st] == 0)
		return;
	for (int i = 1; i <= 8; i++)
		for (int j = 1; j <= 8; j++) {
			chess[i][j] = save[num].chess[i][j][st];
		}
	blackp = save[num].black, whitep = save[num].white;
	scount = st;
	if (st % 2 == 0)
		p = true;
	else
		p = false;

	start();
	Sleep(5000);
	cover();
	return;
}

void setting() {
	putimage(0,0,&setcover);
	drawAlpha(&backbutton,700,700);
	while(1){
		M=getmessage();
		if(M.lbutton==1){
			if(M.x>=20&&M.x<=20+leng&&M.y>=170&&M.y<=170+leng)difficulty=1;
			if(M.x>=140&&M.x<=140+leng&&M.y>=170&&M.y<=170+leng)difficulty=2;
			if(M.x>=260&&M.x<=260+leng&&M.y>=170&&M.y<=170+leng)difficulty=3;
			if(M.x>=600&&M.x<=600+leng&&M.y>=170&&M.y<=170+leng){
				if(bgm)bgm=false;
					else bgm=true;
			}
			if(M.x>=700&&M.x<=700+leng&&M.y>=700&&M.y<=700+leng)cover();
		}
			
		if(difficulty==1)drawAlpha(&ezon,20,170);
			else drawAlpha(&ezoff,20,170);
			
		if(difficulty==2)drawAlpha(&mdon,140,170);
			else drawAlpha(&mdoff,140,170);
			
		if(difficulty==3)drawAlpha(&hdon,260,170);
			else drawAlpha(&hdoff,260,170);
			
		if(bgm)drawAlpha(&bgmon,600,170);
			else drawAlpha(&bgmoff,600,170);
	}
	
	return;
}
