#include <stdio.h>
#include <windows.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
// Guard for older compilers/headers
#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif



int enable_ansi_codes() {
    // 1. Get the handle for standard output (stdout)
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE) {
        return 0; // Failed to get handle
    }

    // 2. Retrieve the current console mode
    DWORD dwMode = 0;
    if (!GetConsoleMode(hOut, &dwMode)) {
        return 0; // Failed to get mode
    }

    // 3. Enable the Virtual Terminal Processing flag
    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    if (!SetConsoleMode(hOut, dwMode)) {
        return 0; // Failed to set mode (system might not support it)
    }

    return 1; // Success!
}

int main() {
    if (enable_ansi_codes()) {
        int i,j,k,l,m,n,frame;
        char square [] = "[_]";
        char grid [10][20][4];
        for (i=0;i<10;i++){
            for (j=0;j<20;j++){
                for (k=0;k<4;k++){
                    grid[i][j][k] = square[k];
                }
            }
        }
        srand(time(NULL));
        FILE* fptr;
        fptr = fopen("input.txt","r");
        if(fptr == NULL) {
        fptr = fopen("filename.txt", "w");
        fclose(fptr);
        fptr = fopen("input.txt","r");
        }else{
        char input[50];
        while(fgets(input,50,fptr)){
        }

        int variables[5] = {0,0,0,0,0};
        //get distance, percentage of cases, transmission rate, percentage of cells, route length
        j=0;
        for (i=0;i<50;i++){
            if (input[i] != '/' && input[i] != NULL){
                k = variables[j];
                variables[j] = k*10 + (int)input[i] - '0';
            }else {
                j++;
            }
        }

        //status check
        for (i=0;i<5;i++){
            printf("%d\n",variables[i]);
        }
        printf("%s\n",input);

        //rand genrator
        int buffer[200 * variables[3] / 100][variables[4]+2];
        for(i=0;i < 200 * variables[3] / 100;i++){
            buffer[i][0] = rand()% 200 + 1;
            for (j=0;j<i;j++){
                 if (buffer[j][0] == buffer[i][0]){
                    i--;
                 }
             }
         }

        for(i=0;i < 200 * variables[3] / 100;i++){
            for(j=1;j < variables[4]+1;j++){
                buffer[i][j] = rand()% 4 + 1;
            }
         }


         for(i=0;i < 200 * variables[3] / 100;i++){
            buffer[i][variables[4]+1] = 0;
         }

         for(i=0;i < (200 * variables[3] / 100)*variables[1]/100;i++){
            buffer[i][variables[4]+1] = 1;
         }

        //status check
        /*printf("%d\n",buffer[0][1]);
        for(i=0;i<200 * variables[3] / 100;i++){
                for(j=0;j<variables[4]+2;j++){
                    printf("%d\t",buffer[i][j]);
                }
            printf("\n");
        }*/

        //routes

        float e = 0;
        int ph;
        frame = 0;
        int area[8];
        //loop
        while(1 == 1){
        scanf("%d",&ph);
        printf("%d\n",ph);
        if(ph == 1){
            for (i=0;i<10;i++){
                for (j=0;j<20;j++){
                    for (l=0;l<(200 * variables[3] / 100);l++){
                        if (i*20 + j == buffer[l][0]-1){
                                if(buffer [l][variables[4]+1] == 0){
                                    for(k=0;k<4;k++){
                                        printf("\033[32m%c",grid[i][j][k]);
                                        }
                                    k++;
                                    break;
                                }else{
                                    for(k=0;k<4;k++){
                                        printf("\033[31m%c",grid[i][j][k]);
                                        }
                                    k++;
                                    break;
                                }
                            }
                        }
                    if(k == 5){
                    k = 0;
                    continue;
                    }else{
                        for(k=0;k<4;k++){
                        printf("\033[37m%c",grid[i][j][k]);
                        }
                   }
                }
               printf("\n");
               }
            printf("\033[37m ");

            m = frame%(2*(variables[4]+1));
            for(i=0;i<(200 * variables[3] / 100);i++){
                if(m < variables[4]+1){
                    switch(buffer[i][m+1]){
                // left
                case 1:
                    n = buffer[i][0] / 20;
                    if((buffer[i][0]-1)/20 == n){
                        buffer[i][0] = buffer[i][0]-1;
                    }else{
                        buffer[i][0] = buffer[i][0] + 19;
                    }
                    break;
                //right
                case 2:
                    n = buffer[i][0] / 20;
                    if((buffer[i][0]+1)/20 == n){
                        buffer[i][0] = buffer[i][0]+1;
                    }else{
                        buffer[i][0] = buffer[i][0] - 19;
                    }
                    break;
                //up
                case 3:
                    if((buffer[i][0]-20 < 0)){
                        buffer[i][0] = buffer[i][0] + 180;
                    }else{
                        buffer[i][0] = buffer [i][0] -20;
                    }
                    break;
                //down
                case 4:
                    if((buffer[i][0]+20 > 200)){
                        buffer[i][0] = buffer[i][0] -180;
                    }else{
                        buffer[i][0] = buffer [i][0] +20;
                    }
                    break;

                }
            }else{
                switch(buffer[i][2*(variables[4]+1) - m]){
                case 1:
                    n = buffer[i][0] / 10;
                    if((buffer[i][0]+1)/10 == n){
                        buffer[i][0] = buffer[i][0]+1;
                    }else{
                        buffer[i][0] = buffer[i][0] - 19;
                    }
                    break;
                case 2:
                    n = buffer[i][0] / 10;
                    if((buffer[i][0]-1)/10 == n){
                        buffer[i][0] = buffer[i][0]-1;
                    }else{
                        buffer[i][0] = buffer[i][0] + 19;
                    }
                    break;
                case 3:
                    if((buffer[i][0]+20 > 200)){
                        buffer[i][0] = buffer[i][0] -180;
                    }else{
                        buffer[i][0] = buffer [i][0] +20;
                    }
                    break;
                case 4:
                    if((buffer[i][0]-20 < 0)){
                        buffer[i][0] = buffer[i][0] + 180;
                    }else{
                        buffer[i][0] = buffer [i][0] -20;
                    }
                    break;
                }

            }


            for(j=0;j<8;j++){
                switch(j){
                case 0:
                    //up
                    if((buffer[i][0] -20 < 0)){
                        area[j] = buffer[i][0] + 180;
                    }else{
                        area[j] = buffer [i][0] -20;
                    }
                    //left
                    n = buffer[i][0] / 20;
                    if((buffer[i][0]-1)/20 == n){
                        area[j] = area[j]-1;
                    }else{
                        area[j] = area[j] + 19;
                    }
                    break;
                case 1:
                    if((buffer[i][0] -20 < 0)){
                        area[j] = buffer[i][0] + 180;
                    }else{
                        area[j] = buffer [i][0] -20;
                    }
                    break;
                case 2:
                    if((buffer[i][0] -20 < 0)){
                        area[j] = buffer[i][0] + 180;
                    }else{
                        area[j] = buffer [i][0] -20;
                    }
                    //right
                    n = buffer[i][0] / 20;
                    if((buffer[i][0]+1)/20 == n){
                        area[j] = area[j]+1;
                    }else{
                        area[j] = area[j] - 19;
                    }
                    break;
                case 3:
                    n = buffer[i][0] / 20;
                    if((buffer[i][0]-1)/20 == n){
                        area[j] = buffer[i][0]-1;
                    }else{
                        area[j] = buffer[i][0] + 19;
                    }
                    break;
                case 4:
                    n = buffer[i][0] / 20;
                    if((buffer[i][0]+1)/20 == n){
                        area[j] = buffer[i][0]+1;
                    }else{
                        area[j] = buffer[i][0] - 19;
                    }
                    break;
                case 5:
                    // down
                    if((buffer[i][0]+20 > 200)){
                        area[j] = buffer[i][0] -180;
                    }else{
                        area[j] = buffer [i][0] +20;
                    }
                    n = buffer[i][0] / 20;
                    if((buffer[i][0]-1)/20 == n){
                        area[j] = area[j]-1;
                    }else{
                        area[j] = area[j] + 19;
                    }
                    break;
                case 6:
                    if((buffer[i][0]+20 > 200)){
                        area[j] = buffer[i][0] -180;
                    }else{
                        area[j] = buffer [i][0] +20;
                    }
                    break;
                case 7:
                    if((buffer[i][0]+20 > 200)){
                        area[j] = buffer[i][0] -180;
                    }else{
                        area[j] = buffer [i][0] +20;
                    }
                    n = buffer[i][0] / 20;
                    if((buffer[i][0]+1)/20 == n){
                        area[j] = area[j]+1;
                    }else{
                        area[j] = area[j] - 19;
                    }
                    break;
                }
            }


            n = 0;
            m = 0;
            for (j=0;j<8;j++){
                if(buffer[area[j]][variables[4]+1] == 1){
                    n++;
                }
            }

            if(n==0){
                continue;
            }else{
                for(k=0;k<n;k++){
                e = (100 - variables[2])/100;
                m = m+pow(e,k)*variables[2];
            }

            n = rand () % 100 + 1;
            if(n < m){
                buffer[i][variables[4]+1] = 1;
            }else{
                continue;
            }
            }
            frame++;
            }

            }else{
                return 0;
            }
        }
        }

       }else {
           printf("Failed to enable ANSI escape codes. Legacy Windows console detected.\n");
       }
   return 0;
}
