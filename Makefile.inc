
#-------------------------------------------------------------
# Select the correct SIMD for your architecture
#-------------------------------------------------------------
#  SIMD Availables:
#      [*] AVX2
#      [*] RVV1_M2_256
#      [*] NEON
#      [*] SVE_128
#      [*] SVE_256
#
#-------------------------------------------------------------
SIMD=RVV1_M2_256
#-------------------------------------------------------------


#------------------------------------------------------------
# Enable Google Highway: 
#   [*] Enable:  Please, introduce the path to the library:
#                GO_HIGHWAY_HOME=<library_path>
#		 
#   [*] Disable: Leave the variable blank.
#                GO_HIGHWAY_HOME=                
#-------------------------------------------------------------
# GO_HIGHWAY_HOME=/home/martineh/software/highway/
  GO_HIGHWAY_HOME=$(HOME)/highway
#-------------------------------------------------------------
