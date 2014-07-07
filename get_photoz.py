import sqlcl
import pandas as pd
import numpy as np
from astropy.coordinates import ICRS

#ra,dec = np.loadtxt('../estimated_redshifts2.tab',dtype='string',usecols=(0,1),unpack=True)
def query_galaxies(ra,dec):
    gal_sdss_data = []
    for i in range(ra.size):
        ra1 = ra[i].split(':')
        dec1 = dec[i].split(':')
        coord = ICRS(ra1[0]+'h'+ra1[1]+'m'+ra1[2]+'s '+dec1[0]+'d'+dec1[1]+'m'+dec1[2]+'s')
        print coord.ra.deg
        print coord.dec.deg
        query = sqlcl.query("SELECT  gn.objid, ISNULL(s.specobjid,0) AS specobjid, p.ra, p.dec,p.Petromag_u-p.extinction_u AS U_mag,p.Petromag_g-p.extinction_g AS G_mag,p.Petromag_r-p.extinction_r AS R_mag,p.Petromag_i-p.extinction_i AS I_mag,p.Petromag_z-p.extinction_z AS Z_mag, ISNULL(s.z, 0) AS z, ISNULL(pz.z, 0) AS pz, 12 FROM  (Galaxy AS p JOIN dbo.fGetNearbyObjEq("+str(coord.ra.deg)+","+str(coord.dec.deg)+","+str(0.033)+") AS GN  ON p.objID = GN.objID LEFT OUTER JOIN SpecObj s ON s.bestObjID = p.objID) LEFT OUTER JOIN Photoz pz on pz.objid = p.objid WHERE p.Petromag_r-p.extinction_r < 19.1 and (status & dbo.fPhotoStatus('GOOD') > 0)").readlines()
        if len(query) > 2:
            print 'oops! More than 1 candidate found'
        if len(query) == 1:
            print 'No targets found'
            gal_sdss_data.append([0,0,0,0,0,0,0,0,0,0,0,0])
            continue
        gal_sdss_data.append(map(float,query[1][:-1].split(',')))
        print 'Done with galaxy',i

    gal_sdss_data = np.array(gal_sdss_data)
    S_df = pd.DataFrame(gal_sdss_data,columns=['#objID','SpecObjID','ra','dec','umag','gmag','rmag','imag','zmag','spec_z','photo_z','extra'])
    return S_df


