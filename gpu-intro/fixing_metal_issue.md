## This is a quick summary of how I fixed an issue with Apple Metal

While completing the introductory tutorial on GPU programming with Mojo, I encountered an error when running the vector addition kernel.

> `error: Metal Compiler failed to compile metallib. Please submit a bug report.`

I resolved this error by:
- Upgrading my OS (to Tahoe 26.1)
- Installing Xcode from the App Store
- Running `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer` in terminal

Now, when I ran `xcrun --sdk macosx --find metal` it returns a path, rather than failing!

And, when I re-run the GPU kernel, it compiles!