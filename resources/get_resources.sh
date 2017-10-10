# LDA
if [ ! -d "./lda-c-dist" ]; then
  wget http://www.cs.columbia.edu/~blei/lda-c/lda-c-dist.tgz
  tar -xzf lda-c-dist.tgz
  rm lda-c-dist.tgz
  cd lda-c-dist
  make 
  cd ..
fi

# word2vec
if [ ! -d "./word2vec" ]; then
  git clone https://github.com/tmikolov/word2vec.git
  cd word2vec
  make
  cd ..
fi
