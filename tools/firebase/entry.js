// Only what DocuLens needs: sign in with Google in a popup, hand back the ID
// token, then forget the Firebase session (the server issues its own login).
import { initializeApp } from "@firebase/app";
import { GoogleAuthProvider, getAuth, signInWithPopup, signOut } from "@firebase/auth";

window.DocuLensFirebase = {
  async signIn(config) {
    const auth = getAuth(initializeApp(config));
    const { user } = await signInWithPopup(auth, new GoogleAuthProvider());
    const token = await user.getIdToken();
    await signOut(auth);
    return token;
  },
};
