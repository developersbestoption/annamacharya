#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Symbol {
    char label[20];
    int addr;
};

struct Symbol symtab[100];
int symcount = 0;

int searchSymbol(char label[]) {
    for (int i = 0; i < symcount; i++) {
        if (strcmp(symtab[i].label, label) == 0)
            return i;
    }
    return -1;
}

void addSymbol(char label[], int addr) {
    strcpy(symtab[symcount].label, label);
    symtab[symcount].addr = addr;
    symcount++;
}

int main() {
    FILE *fp1, *fp2, *fpout;
    char label[20], opcode[20], operand1[20], operand2[20];
    int lc = 0;

    // Pass 1
    fp1 = fopen("input.asm", "r");
    fp2 = fopen("intermediate.txt", "w");

    while (fscanf(fp1, "%s", label) != EOF) {
        if (strcmp(label, "START") == 0) {
            fscanf(fp1, "%s", operand1);
            lc = atoi(operand1);
            fprintf(fp2, "(AD,01) (C,%d)\n", lc);
            continue;
        }

        fscanf(fp1, "%s %s", opcode, operand1);

        if (strcmp(opcode, "END") == 0) {
            fprintf(fp2, "(AD,02)\n");
            break;
        }

        // If label is not '-', store in symbol table
        if (strcmp(label, "-") != 0) {
            if (searchSymbol(label) == -1)
                addSymbol(label, lc);
        }

        if (strcmp(opcode, "DS") == 0) {
            fprintf(fp2, "(DL,02) (C,%s)\n", operand1);
            lc += atoi(operand1);
        } else if (strcmp(opcode, "DC") == 0) {
            fprintf(fp2, "(DL,01) (C,%s)\n", operand1);
            lc++;
        } else {
            fscanf(fp1, "%s", operand2);
            fprintf(fp2, "(IS,%s) (%s) (%s)\n", 
                    strcmp(opcode, "MOVER") == 0 ? "04" :
                    strcmp(opcode, "MOVEM") == 0 ? "05" :
                    strcmp(opcode, "ADD") == 0 ? "01" :
                    strcmp(opcode, "STOP") == 0 ? "00" : "??",
                    strcmp(operand1, "AREG") == 0 ? "1" :
                    strcmp(operand1, "BREG") == 0 ? "2" :
                    strcmp(operand1, "CREG") == 0 ? "3" : "0",
                    operand2);
            lc++;
        }
    }

    fclose(fp1);
    fclose(fp2);

    // Symbol Table Output
    printf("\nSYMBOL TABLE\n");
    printf("Symbol\tAddress\n");
    for (int i = 0; i < symcount; i++)
        printf("%s\t%d\n", symtab[i].label, symtab[i].addr);

    return 0;
}
