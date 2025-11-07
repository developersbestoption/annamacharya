#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_LINES 100
#define MAX_LEN 100

typedef struct {
    char macroName[MAX_LEN];
    int mdtIndex;
} MNTEntry;

typedef struct {
    char line[MAX_LEN];
} MDTEntry;

MNTEntry MNT[MAX_LINES];
MDTEntry MDT[MAX_LINES];

int mntc = 0, mdtc = 0;

// Utility: Trim newline
void trim_newline(char *str) {
    str[strcspn(str, "\n")] = '\0';
}

// Replace &ARG with actual argument
void substitute(char *line, char formal[][MAX_LEN], char actual[][MAX_LEN], int argCount) {
    for (int i = 0; i < argCount; i++) {
        char *pos = strstr(line, formal[i]);
        if (pos != NULL) {
            char temp[MAX_LEN];
            strcpy(temp, line);
            temp[pos - line] = '\0';
            strcat(temp, actual[i]);
            strcat(temp, pos + strlen(formal[i]));
            strcpy(line, temp);
        }
    }
}

int main() {
    FILE *input = fopen("input.asm", "r");
    FILE *output = fopen("output.asm", "w");

    char line[MAX_LEN];
    int inMacro = 0;

    // Pass 1: Build MNT and MDT
    while (fgets(line, sizeof(line), input)) {
        trim_newline(line);

        if (strcmp(line, "MACRO") == 0) {
            inMacro = 1;
            fgets(line, sizeof(line), input);
            trim_newline(line);

            // Parse macro header
            char *token = strtok(line, " ");
            strcpy(MNT[mntc].macroName, token);
            MNT[mntc].mdtIndex = mdtc;

            // Store formal arguments
            char formalArgs[MAX_LINES][MAX_LEN];
            int argCount = 0;
            while ((token = strtok(NULL, ", ")) != NULL) {
                strcpy(formalArgs[argCount++], token);
            }

            // Store macro body into MDT
            while (fgets(line, sizeof(line), input)) {
                trim_newline(line);
                if (strcmp(line, "MEND") == 0) {
                    strcpy(MDT[mdtc++].line, "MEND");
                    break;
                }
                strcpy(MDT[mdtc++].line, line);
            }

            mntc++;
        }
    }

    rewind(input);

    // Pass 2: Expand macros
    while (fgets(line, sizeof(line), input)) {
        trim_newline(line);
        int isMacroCall = 0;

        for (int i = 0; i < mntc; i++) {
            if (strstr(line, MNT[i].macroName) == line) {
                isMacroCall = 1;

                // Parse actual arguments
                char actualArgs[MAX_LINES][MAX_LEN];
                int argCount = 0;
                char *token = strtok(line, " ");
                token = strtok(NULL, ", ");
                while (token != NULL) {
                    strcpy(actualArgs[argCount++], token);
                    token = strtok(NULL, ", ");
                }

                // Expand macro
                int k = MNT[i].mdtIndex;
                while (strcmp(MDT[k].line, "MEND") != 0) {
                    char tempLine[MAX_LEN];
                    strcpy(tempLine, MDT[k].line);

                    // Reconstruct formal args
                    char formalArgs[MAX_LINES][MAX_LEN];
                    for (int a = 0; a < argCount; a++) {
                        sprintf(formalArgs[a], "&ARG%d", a + 1);
                    }

                    substitute(tempLine, formalArgs, actualArgs, argCount);
                    fprintf(output, "%s\n", tempLine);
                    k++;
                }
                break;
            }
        }

        if (!isMacroCall && strcmp(line, "MACRO") != 0 && strcmp(line, "MEND") != 0) {
            fprintf(output, "%s\n", line);
        }
    }

    fclose(input);
    fclose(output);

    printf("Macro processing complete. Check 'output.asm'\n");

    return 0;
}
