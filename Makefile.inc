
#-------------------------------------------------------------
# Select the correct SIMD for your architecture
#-------------------------------------------------------------
#  SIMD Availables:
#      [*] AVX2
#      [*] RVV1
#      [*] NEON
#      [*] SVE_128
#      [*] SVE_256
#
#-------------------------------------------------------------
SIMD=SVE_128
#-------------------------------------------------------------


#------------------------------------------------------------
# Enable Google Highway: 
#   [*] Enable:  Please, introduce the path to the library:
#                GO_HIGHWAY_HOME=<library_path>
#		 
#   [*] Disable: Leave the variable blank.
#                GO_HIGHWAY_HOME=                
#-------------------------------------------------------------
 GO_HIGHWAY_HOME=/home/martineh/software/highway/
# GO_HIGHWAY_HOME=
#-------------------------------------------------------------
