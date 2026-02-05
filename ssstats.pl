#!/usr/bin/perl

open F, 'ssstats.csv' or die();

$matrices = <F>;
$date = <F>;

while (<F>) {
    chomp;
    ($group, $name, $rows, $cols, $nnz,
        $isReal, $isBinary, $isND, $posdef, $psym, $nsym,
        $kind, $nentries) = split ',';

    next if $rows != $cols;
    next if !$isReal || $isBinary;
    next if $kind =~ /graph/i;
    # next if $rows < 10000 || $rows > 100000;
    next if $rows < 100000;
    print "https://suitesparse-collection-website.herokuapp.com/MM/$group/$name.tar.gz\n";
    # print "$group\t$name\t$rows\t$nnz\n"
}
