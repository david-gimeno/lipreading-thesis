#export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/home/dgimeno/kaldi/src/lib:/home/dgimeno/kaldi/tools/openfst-1.6.7/lib

# Defining Kaldi root directory
export KALDI_ROOT=/home/dgimeno/kaldi

# Setting paths to useful tools
export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/nnet3bin/:$PWD:$PATH

# Enable SRILM
sdir=/home/dgimeno/SRILM-1.7.3/bin/i686-m64
export PATH=$PATH:$sdir

# Variable needed for proper data sorting
export LC_ALL=C
