using System;
using System.IO;
using System.Linq;

namespace Lykenox.UtauBridge
{
    class LykenoxUtauBridge
    {
        static int Main(string[] args)
        {
            if (args.Length < 12)
            {
                Console.WriteLine("Usage: lykenox_utau_bridge <input> <output> <pitch> <velocity> <flags> <offset> <length> <consonant> <cutoff> <volume> <modulation> <tempo>");
                return 1;
            }

            try
            {
                string inputWav = args[0];
                string outputWav = args[1];
                string pitch = args[2];
                double velocity = double.Parse(args[3]);
                string flags = args[4];
                double offset = double.Parse(args[5]);
                double length = double.Parse(args[6]);
                double consonant = double.Parse(args[7]);
                double cutoff = double.Parse(args[8]);
                double volume = double.Parse(args[9]);
                double modulation = double.Parse(args[10]);
                double tempo = double.Parse(args[11]);

                Render(inputWav, outputWav, pitch, offset, length, consonant, cutoff, volume);
                return 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error: " + ex.Message);
                return 1;
            }
        }

        static void Render(string input, string output, string pitch, double offset, double length, double consonant, double cutoff, double volume)
        {
            using (var reader = new BinaryReader(File.OpenRead(input)))
            {
                // Skip WAV header (minimal implementation)
                reader.BaseStream.Seek(44, SeekOrigin.Begin);
                byte[] data = reader.ReadBytes((int)(reader.BaseStream.Length - 44));
                short[] samples = new short[data.Length / 2];
                for (int i = 0; i < samples.Length; i++)
                    samples[i] = BitConverter.ToInt16(data, i * 2);

                int sampleRate = 48000;
                int startFrame = (int)(offset * sampleRate / 1000.0);
                int lengthFrames = (int)(length * sampleRate / 1000.0);

                // Minimal UTAU CLI-compatible bridge. This is not OpenUtau WORLDLINE-R.
                short[] result = new short[lengthFrames];
                for (int i = 0; i < lengthFrames; i++)
                {
                    int srcIdx = startFrame + i;
                    if (srcIdx < samples.Length && srcIdx >= 0)
                        result[i] = (short)(samples[srcIdx] * (volume / 100.0));
                }

                // Write output WAV
                using (var writer = new BinaryWriter(File.Create(output)))
                {
                    WriteWavHeader(writer, lengthFrames, sampleRate);
                    for (int i = 0; i < lengthFrames; i++)
                        writer.Write(result[i]);
                }
            }
        }

        static void WriteWavHeader(BinaryWriter writer, int lengthFrames, int sampleRate)
        {
            writer.Write("RIFF".ToCharArray());
            writer.Write(36 + lengthFrames * 2);
            writer.Write("WAVE".ToCharArray());
            writer.Write("fmt ".ToCharArray());
            writer.Write(16);
            writer.Write((short)1); // PCM
            writer.Write((short)1); // Mono
            writer.Write(sampleRate);
            writer.Write(sampleRate * 2);
            writer.Write((short)2);
            writer.Write((short)16);
            writer.Write("data".ToCharArray());
            writer.Write(lengthFrames * 2);
        }
    }
}
