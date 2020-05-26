#!/bin/sh

wget -r -N -c -np --user "$USER" --ask-password https://physionet.org/files/what-is-in-a-note/0.1/
gunzip mimic10.vec.gz
