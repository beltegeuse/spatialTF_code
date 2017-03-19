#include <mitsuba/mitsuba.h>

MTS_NAMESPACE_BEGIN

template <class T>
void writeDensity(Scene* scene, int idPass, int stepDensity, const std::vector<std::vector<PixelData<T> > >& gpsLists) {
   if(((idPass-1) % stepDensity) != 0)
     return;

   Film* film = scene->getFilm();
   ref<Bitmap> bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, film->getSize());
   double totalN = 0.0; //< Normalisation
   for (int i=0; i<(int) gpsLists.size(); ++i) {
     const std::vector<PixelData<T> > &gps = gpsLists[i];
     Spectrum *target = (Spectrum *) bitmap->getUInt8Data();
     for(int j = 0; j < (int)gps.size(); j++) {
	   const PixelData<T> & list = gps[j];
	   Spectrum s(0.f);
	   typename PixelData<T>::const_iterator it = list.begin();
	   for(; it != list.end(); ++it)
	   {
			s += Spectrum(list.N);
			totalN += list.N;
	   }
       target[list.pos.y * bitmap->getWidth() + list.pos.x] = s;
     }
   }
   totalN = std::max(1.0, totalN);
   totalN /= (film->getSize().x*film->getSize().y);

   SLog(EInfo, "Density mult factor: %f", 1.f/totalN);

   for (int i=0; i<(int) gpsLists.size(); ++i) {
     const std::vector<PixelData<T> > &gps = gpsLists[i];
     Spectrum *target = (Spectrum *) bitmap->getUInt8Data();
     for(int j = 0; j < (int)gps.size(); j++) {
       target[gps[j].pos.y * bitmap->getWidth() + gps[j].pos.x] /= Spectrum(totalN);
     }
   }

   std::stringstream ss;
   ss << scene->getDestinationFile().c_str() << "_density_pass_" << idPass;
   std::string path = ss.str();
   fs::path oldName = scene->getDestinationFile();

   film->setBitmap(bitmap);
   film->setDestinationFile(path,0);
   film->develop(scene,0.f);

   // === Revert modification
   film->setDestinationFile(oldName, 0);
 }

MTS_NAMESPACE_END
