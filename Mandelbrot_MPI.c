/* Mandelbrot MPI - UESC / DEC107
 * Estrategias: static (faixas contiguas + MPI_Gatherv) e dynamic
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <stdint.h>

enum { TAG_TASK=10, TAG_RESULT_HEADER=11, TAG_RESULT_DATA=12 };
typedef struct { unsigned char r,g,b; } Cor;
typedef struct {
    int width, height, max_iter, chunk, dynamic;
    double min_real, max_real, min_imag, max_imag;
    const char *output, *raw;
} Config;

static void die(const char *msg, MPI_Comm comm) {
    int rank; MPI_Comm_rank(comm, &rank);
    fprintf(stderr, "[processo %d] %s\n", rank, msg);
    MPI_Abort(comm, 1);
}


static void usage(const char *bin) {
    printf("Uso: %s [--mode static|dynamic] [--width N] [--height N] [--max-iter N]\\n"
           "         [--chunk N] [--center-real X --center-imag Y --span S]\\n"
           "         [--output imagem.ppm] [--raw matriz.bin]\\n"
           "Padrao: -2<=real<=1, -1.5<=imag<=1.5, 4096x4096, max-iter=1000.\\n"
           "Zoom exemplo: --center-real -0.743643887 --center-imag 0.131825904 --span 0.003 --max-iter 5000\\n", bin);
}


static int parse_int(const char *s) {
    char *end; long n = strtol(s, &end, 10);
    if (!*s || *end || n<=0 || n>INT_MAX) return -1;
    return (int)n;
}


static double parse_double(const char *s, MPI_Comm comm) {
    char *end; double n = strtod(s,&end);
    if (!*s || *end || !isfinite(n)) die("Argumento real invalido",comm);
    return n;
}


static Config parse_args(int argc, char **argv, int rank, MPI_Comm comm) {
    Config c={4096,4096,1000,16,0,-2.0,1.0,-1.5,1.5,"mandelbrot.ppm",NULL};
    double cr=0,ci=0,span=0; int got_cr=0,got_ci=0,got_span=0;
    for(int k=1;k<argc;k++) {
        if(!strcmp(argv[k],"--help")) { if(rank==0)usage(argv[0]); MPI_Finalize(); exit(0); }
        if(k+1>=argc)die("Falta valor em argumento",comm);
        const char *key=argv[k], *val=argv[++k];
        if(!strcmp(key,"--mode")) {
            if(!strcmp(val,"static"))c.dynamic=0;
            else if(!strcmp(val,"dynamic"))c.dynamic=1;
            else die("--mode exige static ou dynamic",comm);
        } else if(!strcmp(key,"--width"))c.width=parse_int(val);
        else if(!strcmp(key,"--height"))c.height=parse_int(val);
        else if(!strcmp(key,"--max-iter"))c.max_iter=parse_int(val);
        else if(!strcmp(key,"--chunk"))c.chunk=parse_int(val);
        else if(!strcmp(key,"--center-real")){cr=parse_double(val,comm);got_cr=1;}
        else if(!strcmp(key,"--center-imag")){ci=parse_double(val,comm);got_ci=1;}
        else if(!strcmp(key,"--span")){span=parse_double(val,comm);got_span=1;}
        else if(!strcmp(key,"--output"))c.output=val;
        else if(!strcmp(key,"--raw"))c.raw=val;
        else die("Argumento desconhecido",comm);
    }
    if(c.width<2||c.height<2||c.max_iter<=0||c.chunk<=0)
        die("Dimensoes, max-iter e chunk devem ser positivos",comm);
    if((int64_t)c.width*c.height>INT_MAX)
        die("Quantidade de pixels excede INT_MAX (limite do MPI_Gatherv)",comm);
    if(got_cr||got_ci||got_span) {
        if(!(got_cr&&got_ci&&got_span) || span<=0)
            die("Zoom exige --center-real, --center-imag e --span positivo",comm);
        double half_real=span/2.0;
        double half_imag=span*((double)c.height/c.width)/2.0;
        c.min_real=cr-half_real; c.max_real=cr+half_real;
        c.min_imag=ci-half_imag; c.max_imag=ci+half_imag;
    }
    return c;
}


static void compute_rows(int *out, int first_row, int row_count, const Config *c) {
    const double step_r=(c->max_real-c->min_real)/(c->width-1.0);
    const double step_i=(c->max_imag-c->min_imag)/(c->height-1.0);
    for(int row=first_row;row<first_row+row_count;row++) {
        const double imag=c->min_imag+row*step_i;
        for(int col=0;col<c->width;col++) {
            const double real=c->min_real+col*step_r;
            double zr=0,zi=0;
            int j=0;
            for(;j<c->max_iter;j++) {
                double nr=zr*zr-zi*zi+real;
                double ni=2.0*zr*zi+imag;
                if(nr*nr+ni*ni>4.0) break;
                zr=nr;zi=ni;
            }
            out[(size_t)(row-first_row)*c->width+col]=j;
        }
    }
}


static void map_colors(double v, unsigned char *r,unsigned char *g,unsigned char *b) {
    const Cor colors[]={ {11,29,58},{18,56,110},{31,111,180},
                         {34,182,200},{240,180,41},{255,243,209} };
    if(v<0)v=0;
    if(v>1)v=1;
    double pos=v*5.0; int i=(int)pos; if(i>4)i=4;
    double t=pos-i;
    *r=(unsigned char)(colors[i].r+t*(colors[i+1].r-colors[i].r));
    *g=(unsigned char)(colors[i].g+t*(colors[i+1].g-colors[i].g));
    *b=(unsigned char)(colors[i].b+t*(colors[i+1].b-colors[i].b));
}


static void write_image(const Config *c, const int *pixels, MPI_Comm comm) {
    FILE *f=fopen(c->output,"wb"); if(!f)die("Nao foi possivel abrir PPM",comm);
    fprintf(f,"P6\n%d %d\n255\n",c->width,c->height);
    unsigned char *palette=malloc((size_t)c->max_iter*3);
    unsigned char *line=malloc((size_t)c->width*3);
    if(!palette||!line)die("Falha ao alocar paleta/linha",comm);
    const double log_max=log1p((double)c->max_iter);
    for(int i=0;i<c->max_iter;i++)
        map_colors(log1p((double)i)/log_max,&palette[3*(size_t)i],
                   &palette[3*(size_t)i+1],&palette[3*(size_t)i+2]);
    for(int row=0;row<c->height;row++) {
        for(int col=0;col<c->width;col++) {
            size_t p=(size_t)row*c->width+col;
            int iter=pixels[p];
            if(iter>=c->max_iter) { line[3*(size_t)col]=11;line[3*(size_t)col+1]=29;line[3*(size_t)col+2]=58; }
            else memcpy(&line[3*(size_t)col],&palette[3*(size_t)iter],3);
        }
        if(fwrite(line,3,(size_t)c->width,f)!=(size_t)c->width)die("Erro ao gravar PPM",comm);
    }
    free(line);free(palette);fclose(f);
    if(c->raw) {
        FILE *raw=fopen(c->raw,"wb");if(!raw)die("Nao foi possivel criar arquivo raw",comm);
        size_t count=(size_t)c->width*c->height;
        if(fwrite(pixels,sizeof(int),count,raw)!=count)die("Erro ao gravar matriz raw",comm);
        fclose(raw);
    }
}


static int start_row(int rank,int p,int h) { return (int)((int64_t)rank*h/p); }


static int rows_of(int rank,int p,int h) { return start_row(rank+1,p,h)-start_row(rank,p,h); }


int main(int argc,char **argv) {
    MPI_Init(&argc,&argv);
    MPI_Comm comm=MPI_COMM_WORLD; int rank,p;
    MPI_Comm_rank(comm,&rank);MPI_Comm_size(comm,&p);
    Config c=parse_args(argc,argv,rank,comm);
    const size_t total=(size_t)c.width*c.height;
    int *matrix=NULL;
    if(rank==0){matrix=malloc(total*sizeof(int));if(!matrix)die("Falha alocacao matriz global",comm);}
    double compute=0,comm_time=0;
    int local_rows=0;int *local=NULL;
    if(!c.dynamic || p==1) {
        local_rows=rows_of(rank,p,c.height);
        local=malloc((size_t)(local_rows?local_rows:1)*c.width*sizeof(int));
        if(!local)die("Falha na alocacao local",comm);
    } else if(rank!=0) {
        local=malloc((size_t)c.chunk*c.width*sizeof(int));
        if(!local)die("Falha na alocacao do pacote",comm);
    }
    MPI_Barrier(comm);
    double start=MPI_Wtime();
    if(!c.dynamic || p==1) {
        double t=MPI_Wtime();
        compute_rows(local,start_row(rank,p,c.height),local_rows,&c);
        compute+=MPI_Wtime()-t;
        int *counts=NULL,*displs=NULL;
        if(rank==0) {
            counts=malloc((size_t)p*sizeof(int));displs=malloc((size_t)p*sizeof(int));
            if(!counts||!displs)die("Falha na alocacao Gatherv",comm);
            for(int r=0;r<p;r++){
                counts[r]=rows_of(r,p,c.height)*c.width;
                displs[r]=start_row(r,p,c.height)*c.width;
            }
        }
        t=MPI_Wtime();
        MPI_Gatherv(local,local_rows*c.width,MPI_INT,matrix,counts,displs,MPI_INT,0,comm);
        comm_time+=MPI_Wtime()-t;
        free(counts);free(displs);
    } else if(rank==0) {
        int next=0, active=0;
        for(int worker=1;worker<p;worker++) {
            int task[2];
            if(next<c.height) {task[0]=next;task[1]=(c.height-next<c.chunk)?c.height-next:c.chunk;next+=task[1];active++;}
            else {task[0]=-1;task[1]=0;}
            double t=MPI_Wtime();MPI_Send(task,2,MPI_INT,worker,TAG_TASK,comm);comm_time+=MPI_Wtime()-t;
        }
        while(active>0) {
            int hdr[2]; MPI_Status st;
            double t=MPI_Wtime();
            MPI_Recv(hdr,2,MPI_INT,MPI_ANY_SOURCE,TAG_RESULT_HEADER,comm,&st);
            int worker=st.MPI_SOURCE;
            MPI_Recv(matrix+(size_t)hdr[0]*c.width,hdr[1]*c.width,
                     MPI_INT,worker,TAG_RESULT_DATA,comm,MPI_STATUS_IGNORE);
            comm_time+=MPI_Wtime()-t;
            int task[2];
            if(next<c.height){task[0]=next;task[1]=(c.height-next<c.chunk)?c.height-next:c.chunk;next+=task[1];}
            else {task[0]=-1;task[1]=0;active--;}
            t=MPI_Wtime();MPI_Send(task,2,MPI_INT,worker,TAG_TASK,comm);comm_time+=MPI_Wtime()-t;
        }
    } else {
        for(;;) {
            int task[2];double t=MPI_Wtime();
            MPI_Recv(task,2,MPI_INT,0,TAG_TASK,comm,MPI_STATUS_IGNORE);
            comm_time+=MPI_Wtime()-t;
            if(task[0]<0)break;
            t=MPI_Wtime();compute_rows(local,task[0],task[1],&c);compute+=MPI_Wtime()-t;
            t=MPI_Wtime();
            MPI_Send(task,2,MPI_INT,0,TAG_RESULT_HEADER,comm);
            MPI_Send(local,task[1]*c.width,MPI_INT,0,TAG_RESULT_DATA,comm);
            comm_time+=MPI_Wtime()-t;
        }
    }
    /* Todos concluem a geracao antes do relogio parar; escrita nao esta incluida. */
    MPI_Barrier(comm);
    double total_time=MPI_Wtime()-start;
    double *computes=NULL,*comms=NULL;
    if(rank==0){computes=malloc((size_t)p*sizeof(double));comms=malloc((size_t)p*sizeof(double));
        if(!computes||!comms)die("Falha alocacao estatisticas",comm);}
    MPI_Gather(&compute,1,MPI_DOUBLE,computes,1,MPI_DOUBLE,0,comm);
    MPI_Gather(&comm_time,1,MPI_DOUBLE,comms,1,MPI_DOUBLE,0,comm);
    if(rank==0) {
        int first=(c.dynamic && p>1)?1:0;
        double min=computes[first],max=computes[first];
        for(int r=first;r<p;r++){
            if(computes[r]<min)min=computes[r];
            if(computes[r]>max)max=computes[r];
        }
        printf("Modo=%s processos=%d tamanho=%dx%d max_iter=%d chunk=%d\n",
               c.dynamic?"dynamic":"static",p,c.width,c.height,c.max_iter,c.chunk);
        printf("Tempo_geracao_s=%.6f (inclui coleta e sincronizacao; exclui escrita)\n",total_time);
        printf("Processo | Computacao_s | Comunicacao_MPI_s\n");
        for(int r=0;r<p;r++)printf("%8d | %12.6f | %17.6f%s\n",r,computes[r],comms[r],
                               (c.dynamic&&p>1&&r==0)?" (coordenador)":"");
        printf("FBC=%.6f (T = somente computacao, %s)\n",max>0?(max-min)/max:0,
               (c.dynamic&&p>1)?"exclui coordenador":"inclui todos os processos");
        double io=MPI_Wtime(); write_image(&c,matrix,comm);io=MPI_Wtime()-io;
        printf("Tempo_escrita_s=%.6f\n",io);
        free(computes);free(comms);free(matrix);
    }
    free(local);
    MPI_Finalize();return 0;
}
