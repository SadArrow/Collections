//#include <easyxgraphics.h>
#include <easyx.h>
#include <graphics.h> 
#include <conio.h>
#include <bits/stdc++.h>
#include <Windows.h>
#include <mmsystem.h>
#pragma comment(lib,"winmm.lib")
#define float double
using namespace std;

#include "initialize.h"
#include "graph.h"
#include "AI_ez.h"
#include "AI_medium.h"
#include "AI_hard.h"
#include "game.h"

int main() {
	filein(); 
	initialize();
	cover();
	getch();
	closegraph();

	return 0;
}

