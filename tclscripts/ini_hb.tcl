mol delrep 0 0

# Set all
mol color Name
mol representation Lines 4.000000
mol selection all
mol material Opaque
mol addrep 0

# Set Strand 1, 5'-end
mol color ColorID 11
mol representation VDW 0.700000 12.000000
mol selection fragment 0 and resid 1 and name O5'
mol material Opaque
mol addrep 0

# Set Strand 1, 3'-end
mol color ColorID 2
mol representation VDW 0.700000 12.000000
mol selection fragment 0 and resid 21 and name O3'
mol material Opaque
mol addrep 0

# Set Strand 2, 5'-end
mol color ColorID 12
mol representation VDW 0.700000 12.000000
mol selection fragment 1 and resid 22 and name O5'
mol material Opaque
mol addrep 0

# Set Strand 2, 3'-end
mol color ColorID 16
mol representation VDW 0.700000 12.000000
mol selection fragment 1 and resid 42 and name O3'
mol material Opaque
mol addrep 0

