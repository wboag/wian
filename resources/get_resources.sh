#!/bin/sh

wget --user "$USER" --ask-password https://physionet.org/works/MIMICIIIDerivedDataRepository/files/approved/what-is-in-a-note/mimic10.vec.gz
gunzip mimic10.vec.gz
